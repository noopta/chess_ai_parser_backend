use std::process::{Stdio};

use tokio::io::split;
use tokio::io::{AsyncWriteExt, AsyncBufReadExt, BufReader};
use tonic::{transport::Server, Request, Response, Status};
use reqwest::header;
use reqwest::Client;
use std::sync::{Arc, PoisonError};
use tokio::sync::{Mutex, mpsc::Receiver};
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use anyhow::{Result, Error};
use tonic_web::GrpcWebLayer;
use tower_http::cors::{CorsLayer, Any, AllowOrigin};
use tonic::codegen::http::HeaderValue;

// use hello_world::greeter_server::{Greeter, GreeterServer};

use regex::Regex;
use tokio::sync::oneshot;
use chess::{Board, ChessMove, Piece, Square};
use std::str::FromStr;
use std::time::Instant;
use tokio::{
    process::Command
};
use tokio::sync::mpsc::{self, Sender};
use tower::ServiceBuilder;

pub mod hello_cargo {
    tonic::include_proto!("hello_cargo");
}

use hello_cargo::greeter_server::{Greeter};
use hello_cargo::{ChessApiGameResponse, ParsedChessGameData, ChessGamesResponse, HelloReply, HelloRequest};
use hello_cargo::chess_service_server::{ChessService, ChessServiceServer};
use hello_cargo::{ProfileRequestData};

#[derive(Debug)]
pub struct AnalysisRequest {
    pub lan_moves: Vec<String>,
    pub response_tx: oneshot::Sender<AnalysisResult>,
}

#[derive(Debug)]
pub struct AnalysisResult {
    pub probabilities: Vec<f64>,
    pub best_moves: Vec<String>,
    pub fen_values: Vec<String>,
}

#[derive(Debug)]
pub struct StockfishPool {
    tx: Sender<AnalysisRequest>,
}

impl StockfishPool {
    pub fn new(num_instances: usize) -> Self {
        let (tx, rx) = mpsc::channel::<AnalysisRequest>(1000);
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..num_instances {
            let rx_shared = Arc::clone(&rx);

            tokio::spawn(async move {
                let _ = create_stockfish_instance(rx_shared).await;
            });
        }

        drop(rx);

        StockfishPool {tx}
    }

    pub async fn analyze_game(
        &self,
        lan_moves: Vec<String>,
    ) -> anyhow::Result<AnalysisResult> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let req = AnalysisRequest {
            lan_moves,
            response_tx,
        };

        // send the request to the pool

        self.tx.send(req).await
            .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

        let result = response_rx
            .await
            .map_err(|e| anyhow::anyhow!("Worker dropped: {}", e))?;

        println!("result received: {} {} { }", result.best_moves.len(), result.fen_values.len(), result.probabilities.len());
        Ok(result)
    }
}

pub async fn create_stockfish_instance(mut rx: Arc<Mutex<Receiver<AnalysisRequest>>>) -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new("../Stockfish/src/stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

    let mut child_stdin = child.stdin.take().unwrap();
    let mut child_stdout = BufReader::new(child.stdout.take().unwrap());

    child_stdin.write_all(b"uci\n").await?;
    child_stdin.write_all(b"isready\n").await?;
    child_stdin.flush().await?;

    {
        let mut line = String::new();
        while child_stdout.read_line(&mut line).await? > 0 {
            if line.contains("uciok") || line.contains("readyok") {
                break;
            }
            line.clear();
        }
    }

    while let Some(req) = {

        let mut locked_rx = rx.lock().await;

        locked_rx.recv().await
    } {
        // We'll replicate your "move-by-move" logic here,
        // but for each game in a single request.

        let mut probabilities: Vec<f64> = Vec::new();
        let mut best_moves: Vec<String> = Vec::new();
        let mut fen_values: Vec<String> = Vec::new();
        let mut current_moves = String::new();

        // For each move in that game, feed it to Stockfish
        for mv in &req.lan_moves {
            // Append this move to the running move list
            if !current_moves.is_empty() {
                current_moves.push(' ');
            }
            current_moves.push_str(mv);

            // Convert to FEN if you want to store it
            let fen = convert_lan_to_fen(&current_moves); 
            fen_values.push(fen);

            // position startpos moves ...
            let position_cmd = format!("position startpos moves {}\n", current_moves);
            child_stdin.write_all(position_cmd.as_bytes()).await?;

            // go depth 6 (or whatever)
            child_stdin.write_all(b"go depth 6\n").await?;
            child_stdin.flush().await?;

            // Now read lines from child_stdout until we see "bestmove"
            // collecting the CP or probability from lines starting with "info depth 6"
            let (cp_chance, best) = read_one_move_analysis(&mut child_stdout).await?;
            probabilities.push(cp_chance);
            best_moves.push(best);
        }

        // We have all data for this game. Construct an AnalysisResult
        let analysis_result = AnalysisResult {
            probabilities,
            best_moves,
            fen_values,
        };

        // Send result back to the caller
        if let Err(analysis_that_failed_to_send) = req.response_tx.send(analysis_result) {
            // The receiver was dropped, so no one is expecting the result anymore.
            eprintln!("Failed to send AnalysisResult because the receiver was dropped.");
        }
    }
        // 3) If channel is closed, we quit
        child_stdin.write_all(b"quit\n").await?;
        child_stdin.flush().await?;
        let _ = child.wait().await?;
        Ok(())
}


async fn read_one_move_analysis(
    stdout: &mut BufReader<tokio::process::ChildStdout>,
) -> anyhow::Result<(f64, String)> {
    let mut cp_chance = 0.0;
    let mut best_move = String::new();
    let mut line = String::new();

    while stdout.read_line(&mut line).await? > 0 {
        // e.g. lines with "info depth 6 cp <val>" 
        if line.starts_with("info depth 6") {
            let split_line: Vec<&str> = line.split_whitespace().collect();
            // find "cp" and parse next
            if let Some(cp_idx) = split_line.iter().position(|s| *s == "cp") {
                if cp_idx + 1 < split_line.len() {
                    if let Ok(cp_value) = split_line[cp_idx + 1].parse::<i32>() {
                        cp_chance = (convert_cp_to_chances(&cp_value) * 100.0).round() / 100.0;
                    }
                }
            }
        } else if line.starts_with("bestmove") {
            // parse out the bestmove token
            let split_line: Vec<&str> = line.split_whitespace().collect();
            if split_line.len() >= 2 {
                best_move = split_line[1].to_string();
            }
            break;
        } else {
            println!("engine: {} ", line);
        }
        line.clear();
    }

    Ok((cp_chance, best_move))
}

struct MovesTuple(f64, i32);

impl PartialEq for MovesTuple {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MovesTuple {}

impl Ord for MovesTuple {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MovesTuple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


fn parse_game_pgn(pgn: &str) -> String {

    let split_pgn: Vec<&str> = pgn.split("\n").collect();

    let game_moves = split_pgn[split_pgn.len() - 2];

    // println!("game move: {}", game_moves.to_string());
    // for line in split_pgn {
    //     println!("Line: {}", line.to_string());
    // }

    "abc".to_string()
}  

fn convert_lan_to_fen(lan_moves: &str) -> String {
    let mut board = Board::default(); // Standard initial position

    for lan_move in lan_moves.split_whitespace() {
        if(lan_move.len() == 0) {
            continue;
        }

        // Parse the LAN string (e.g. "e2e4", "e7e8=Q", "e7e8Q", etc.) into a ChessMove
        let chess_move = parse_lan(lan_move)
            .unwrap_or_else(|err| panic!("Invalid LAN '{}': {}", lan_move, err));
        
        // Optionally, you might want to verify the move is legal in `board`:
        // if !board.is_legal(chess_move) {
        //     panic!("Illegal move '{}' in current position.", lan_move);
        // }

        // Make the move on a new board
        board = board.make_move_new(chess_move);
    }

    // Return the FEN of the resulting position
    board.to_string()
}

/// Parse a LAN string (like "e2e4", "e7e8=Q", "e7e8Q") into a `ChessMove`.
fn parse_lan(lan: &str) -> Result<ChessMove, String> {
    // The first four characters should be from- and to-squares, e.g. "e2e4".
    if lan.len() < 4 {
        return Err("LAN move too short".to_string());
    }

    let from_str = &lan[0..2];
    let to_str   = &lan[2..4];

    // Convert strings like "e2" into a `Square`
    let from_square = Square::from_str(from_str)
        .map_err(|_| format!("Invalid 'from' square: {}", from_str))?;
    let to_square = Square::from_str(to_str)
        .map_err(|_| format!("Invalid 'to' square: {}", to_str))?;

    // Check for a promotion piece (Q, R, B, N) with or without '='.
    // Examples of valid promotions: "e7e8=Q", "e7e8Q".
    let promotion = match lan.len() {
        5 => {
            // e.g. "e7e8Q"
            let promo_char = lan.chars().nth(4).unwrap();
            Some(char_to_piece(promo_char)?)
        }
        6 if lan.chars().nth(4) == Some('=') => {
            // e.g. "e7e8=Q"
            let promo_char = lan.chars().nth(5).unwrap();
            Some(char_to_piece(promo_char)?)
        }
        _ => None, // No promotion
    };

    // Construct the ChessMove
    Ok(ChessMove::new(from_square, to_square, promotion))
}

/// Helper to convert a char ('Q', 'R', 'B', 'N') into the corresponding `Piece`.
fn char_to_piece(c: char) -> Result<Piece, String> {
    match c {
        'Q' => Ok(Piece::Queen),
        'R' => Ok(Piece::Rook),
        'B' => Ok(Piece::Bishop),
        'N' => Ok(Piece::Knight),
        _ => Err(format!("Invalid promotion piece: '{}'", c)),
    }
}


fn convert_pgn_moves_to_lan(moves: Vec<String>) -> Vec<String> {
    let mut board = Board::default();
    let mut lan_moves = Vec::new();

    for curr_move in moves {
        if let Ok(chess_move) = ChessMove::from_san(&board, &curr_move) {
            let origin = chess_move.get_source();
            let dest = chess_move.get_dest();

            let lan_move = format!("{}{}", origin, dest);
       
            lan_moves.push(lan_move);

            board = board.make_move_new(chess_move);
        } else {
            println!("Invalid move: {}", curr_move);
        }
    }

    lan_moves
}

fn convert_cp_to_chances(cp: &i32) -> f64{
    let float_cp: f64 = *cp as f64;

    let exponent: f64 = -1.0 * (float_cp / 700.0);

    // println!("exponent = {}", exponent.to_string());

    let base: f64 = 10.0;

    return 1.0 / (1.0 + ((base.powf(exponent))));
}

fn convert_to_san(input: &str) -> String {
    // --- 1. Collect only lines that contain actual moves ---
    // We can do this by checking for lines that start with a digit and a period ("1.", "2.", etc.)
    // or by detecting something that looks like moves.

    // Split everything into lines
    let lines: Vec<&str> = input.lines().collect();
    let mut moves_lines = Vec::new();
    for line in lines {
        // Trim whitespace
        let line = line.trim();
        // Simple heuristic: keep lines that start with "<digit>."
        if line.starts_with(|ch: char| ch.is_digit(10)) {
            moves_lines.push(line);
        }
    }

    // --- 2. Combine all those lines into one big string (PGN might have multiple lines) ---
    let combined_moves = moves_lines.join(" ");

    // --- 3. Use a regex to find sequences like:
    //         N... or O-O, or a-h files, +, #, capture 'x', or number dots, ignoring annotations.
    // A simpler approach is to first remove clock annotations like {[%clk xx:xx]}.
    let annotation_regex = Regex::new(r"\{[^}]*\}").unwrap(); // Matches { ... } blocks
    let no_annotations = annotation_regex.replace_all(&combined_moves, "");

    // Remove move numbers like "1." or "1..." etc.
    let move_number_regex = Regex::new(r"\d+\.+").unwrap();
    let no_moveno = move_number_regex.replace_all(&no_annotations, "");

    // Now we should have something like: "e4 e5 Nc3 Nf6 b3 Nc6 Bb2 d6 Qe2 g6 Nd5 ..."
    // Let's split on whitespace to get the tokens
    let tokens: Vec<&str> = no_moveno.split_whitespace().collect();

    // --- 4. Merge them into pairs: (WhiteMove, BlackMove) => "e4e5, Nc3Nf6, ..."
    // Actually, because the PGN can have an odd number of tokens (some partial moves), we should handle that carefully.

    let mut output_moves = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        // White move
        let white_move = tokens[i];
        // println!("token: {}", tokens[i]);
        i += 1;

        // Black move (if exists)
        if i < tokens.len() {
            let black_move = tokens[i];
            i += 1;
            output_moves.push(format!("{}{}", white_move, black_move));
        } else {
            // Odd number, so last move has no pair
            output_moves.push(white_move.to_string());
        }
    }

    // --- 5. Join moves with ", " ---
    // let final_output = output_moves.join(", ");
    let final_output = tokens.join(", ");
    final_output
}

#[derive(Debug, Default)]
pub struct MyGreeter {}

#[derive(Debug)]
pub struct ChessStruct{
    pool: StockfishPool,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct LocalChessApiGameResponse {
    games: Vec<LocalGame>
}

#[derive(serde::Deserialize, Debug, Clone)]
struct LocalGame {
    url: String,
    pgn: String,
    tcn: String,
    fen: String,
    white: LocalChessPlayer,
    black: LocalChessPlayer
}

#[derive(serde::Deserialize, Debug, Clone)]
struct LocalChessPlayer {
    rating: i32,
    result: String,
    username: String
}

impl LocalChessApiGameResponse {
    fn into_proto(self) -> hello_cargo::ChessApiGameResponse {
        hello_cargo::ChessApiGameResponse {
            // map local games into proto games
            games: self.games.into_iter().map(|g| g.into_proto()).collect(),
        }
    }
}

impl LocalGame {
    fn into_proto(self) -> hello_cargo::Game {
        hello_cargo::Game {
            url: self.url,
            pgn: self.pgn,
            tcn: self.tcn,
            fen: self.fen,
            white: Some(self.white.into_proto()),
            black: Some(self.black.into_proto()),
        }
    }
}

impl LocalChessPlayer {
    fn into_proto(self) -> hello_cargo::ChessPlayer {
        hello_cargo::ChessPlayer {
            rating: self.rating,
            result: self.result,
            username: self.username,
        }
    }
}



async fn fetch_games(username: &str, month: &str, year: &str) -> Result<LocalChessApiGameResponse, Box<dyn std::error::Error + Send + Sync>> {
    let _endpoint_url: String = format!(
        "https://api.chess.com/pub/player/{}/games/{}/{}",
        username, year, month
    );

   // Build a client so we can add headers
   let client = Client::new();
   let response = client
       .get(&_endpoint_url)
       // Add a custom user agent
       .header(header::USER_AGENT, "wallace.dev@proton.me")
       .send()
       .await?;


    // let response_text = reqwest::get(&_endpoint_url).await?;

    // let chess_games: Vec<ChessApiGameResponse> = serde_json::from_str(&response_text)?;
    // let parsed_games: Vec<Game> = chess_games.into_iter().map(convert_game).collect();
    
    if response.status().is_success() {
        let json_response: LocalChessApiGameResponse = response.json().await?;
        Ok(json_response)
    } else {
        println!("HTTP REQUEST FAILED WITH SSTATUS: {}", response.status());
        Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "HTTP REQUEST FAILED",
        )))
    }

    // for (idx, game) in parsed_games.iter().enumerate() {
    //     println!("game: {} index: {}", game.url, idx);
    // }

    // Ok(ChessApiResponse{games: parsed_games})
}


// so currently we will get a list of games back 
// each game will have it's own FEN, SAN, best_moves, etc
// so therefore, we need a struct that can hold this list of game information


impl ChessStruct {
    pub fn new() -> Self {
        let pool = StockfishPool::new(2);
        ChessStruct {pool}
    }
}

// chessService is a trait which is basically an abstract class 
#[tonic::async_trait]
impl hello_cargo::chess_service_server::ChessService for ChessStruct {
    async fn get_chess_games(
        &self,
        request: Request<hello_cargo::ProfileRequestData>,
    ) -> Result<Response<hello_cargo::ChessGamesResponse >, Status> {
        let start = Instant::now();
        let request_ref = request.get_ref().clone();
        println!("Chess request {:?}", request_ref);
        
        // int rating = 1;
        // string result = 2;
        // string username = 3;
        let content = request.get_ref();
        // let innerContent = request.into_inner().clone();

        let username = &content.username;
        let month = &content.month;
        let year = &content.year;
        let mut proto_response = hello_cargo::ChessGamesResponse::default();

        let mut chess_proto_response: Arc<Mutex<ChessGamesResponse>> = Arc::new(Mutex::new(ChessGamesResponse::default()));
        let mut chess_proto_response_thread: Arc<Mutex<ChessGamesResponse>> = Arc::clone(&chess_proto_response);


        // let mut probabilities: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        // let probabilities_thread: Arc<Mutex<Vec<f64>>> = Arc::clone(&probabilities);

        match fetch_games(username, month, year).await {
            Ok(chess_response) => {
                // println!("Fetched chess response: {:?}", chess_response); // Print the raw fetched data
                
                // let request_ref = request.get_ref().clone();


                // so the way stockfish works is that it takes in a FEN and recommends the next best move from there 
                // so.. one way is to get every individual fen in the game 

                // is there another way?

                let games = chess_response.games.clone();

                let game_proto_response = Arc::new(Mutex::new(ChessGamesResponse::default()));

                for first_game in games {

                    let chess_san = convert_to_san(&first_game.pgn);

                    let split_moves = chess_san
                        .split(", ")
                        .map(|s| s.to_string())
                        .collect();
    
                    let lan_moves: Vec<String> = convert_pgn_moves_to_lan(split_moves);
                    let lan_moves_copy = lan_moves.clone();


                    let analysis = match self.pool.analyze_game(lan_moves_copy).await {
                        Ok(res) => {
                            res
                        },
                        Err(e) => {
                            return Err(Status::internal(format!("Analysis error: {e}")));
                        }
                    };

                    let mut parsed_game_data = hello_cargo::ParsedChessGameData::default();
                    parsed_game_data.fen_values = analysis.fen_values;
                    parsed_game_data.lan_moves = lan_moves;
                    parsed_game_data.probabilities = analysis.probabilities;
                    parsed_game_data.best_moves = analysis.best_moves;
                    parsed_game_data.username = &username;
                    

                    {
                        let mut guard = game_proto_response.lock().await;
                        // println!("best moves len: {} fen len: {} lan moves len: {}", parsed_game_data.best_moves.len(), parsed_game_data.fen_values.len(), parsed_game_data.lan_moves.len());
                        guard.games.push(parsed_game_data);
                    }

                    // let's just focus on creating the functions necessary for one game, 
                    // then ideally we should have the approach for all games at scale
    
                    // here we have a string of san 
                    // we can get the probabilities of both black and white at each point

                }

                // println!("len of chess games: {}", chess_proto_response.lock().unwrap().games.len());

                let guard = game_proto_response.lock().await;
                println!("games: {}", guard.games.len());
                
                let guard_clone = guard.clone();

                let first_game = guard_clone.games.get(0).unwrap();
                

                for i in 0..first_game.best_moves.len() {
                    println!("move: {} best move: {} white's prob curr: {}", first_game.lan_moves[i], first_game.best_moves[i], first_game.probabilities[i]);
                }


                let duration = start.elapsed();
                println!("Program time: {}", duration.as_millis());
                Ok(Response::new(guard.clone()))
            },
            Err(e) => Err(Status::internal(format!("Error fetching games: {}", e))),
        }
        // https://api.chess.com/pub/player/noopdogg07/games/2024/010
        // Ok(Response::new(reply))
    }
}

// OLD CODE 
// let split_san = chess_san.split(", ");
                    
// let mut probabilites: Vec<f64> = Vec::new();

// let mut child = Command::new("../Stockfish/src/stockfish")
// // creates a standard input pipe
//     .stdin(Stdio::piped())
//     // creates a standard output pipe 
//     .stdout(Stdio::piped())
//     .spawn()?; // spawns the child process

// let mut stdin = child.stdin.take().unwrap();
// let stdout = child.stdout.take().unwrap();
// let mut depth_one_cp: f64;
// let mut probabilities: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
// let probabilities_thread: Arc<Mutex<Vec<f64>>> = Arc::clone(&probabilities);

// let mut best_moves: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
// let mut best_moves_thread: Arc<Mutex<Vec<String>>> = Arc::clone(&best_moves);

// let mut fen_values: Vec<String> = Vec::new();

// let reader_thread = std::thread::spawn(move || {
//     let reader = BufReader::new(stdout);

//     for line in reader.lines() {
//         if let Ok(line) = line {
//             if line.starts_with("info 6") {
//                 // println!("ENGINE: {}", line);
//                 let split_line: Vec<&str> = line.split(" ").collect();

//                 for i in 0..split_line.len() {
//                     if split_line[i].to_string().starts_with("cp") {
//                         let parsed_value: i32 = split_line[i + 1].parse().expect("Not a valid integer");
//                         let mut probs = probabilities_thread.lock().unwrap();
                      
//                         // probabilities_thread.push(convert_cp_to_chances(&parsed_value));
//                         // (x * 100.0).round() / 100.0
//                         probs.push((convert_cp_to_chances(&parsed_value) * 100.0).round() / 100.0);
//                         // probs.push(convert_cp_to_chances(&parsed_value));
//                         break;
//                     } 
//                 }
//             } else if line.starts_with("bestmove") {
//                 // println!("ENGINE BEST MOVE: {}", line);
//                 let mut moves = best_moves_thread.lock().unwrap();

//                 moves.push(line);
//             }
//         }
//     }
// });


// let mut current_moves: String = "".to_string();
// let mut count = 3;

// // writeln!(stdin, "position startpos")?;
// // writeln!(stdin, "go depth 10")?;
// writeln!(stdin, "uci")?;
// writeln!(stdin, "isready")?;
// for curr_move in &lan_moves {
//     // feed move by move to stockfish and store the result 
//     // in an array

//     // add the current move to the current_moves
//     // then feed the current_moves string to stockfish
//     // tell stockfish to go a certain depth 

//     // then read the lines back 
    
//     current_moves.push_str(&curr_move);
    
//     fen_values.push(convert_lan_to_fen(&current_moves));
//     println!("FEN: {}", convert_lan_to_fen(&current_moves));
//     println!("current move: {}", curr_move);
//     current_moves.push_str(" ");
//     // writeln!(stdin, "position startpos moves {}", current_moves)?;

//     writeln!(stdin, "position startpos moves {}", current_moves)?;
//     writeln!(stdin, "go depth 6")?;
// }

// writeln!(stdin, "quit")?;

// // println!("game in SAN: {}", convert_to_san(&first_game.pgn));
// let cp: i32 = 100;

// let result = convert_cp_to_chances(&cp);

// // println!("result cp: {}", result.to_string());

// // proto_response = chess_response.into_proto();
// reader_thread.join().expect("Issue closing reader thread");

// let final_probabilities =  probabilities.lock().unwrap();

// // for curr_prob in final_probabilities.iter() {
// //     println!("prob = {}", curr_prob);
// // }
// let mut player: String = "white".to_string();
// let mut white_probs: Vec<f64> = Vec::new();
// let mut black_probs: Vec<f64> = Vec::new();

// for i in 0..final_probabilities.len() {
//     black_probs.push(((1.0 - final_probabilities[i]) * 100.0).round() / 100.0);
// }

// // so this won't work if the other probability + curr > 1 or less than 1
// // so just pick one

// // each probabilty calculated is white's chance to win
// let mut flag = false;

// for i in 0..black_probs.len() {
//     if(!flag) {
//         // println!("white = {} black = {} move = {}", final_probabilities[i], black_probs[i], lan_moves[i]);
//     }
    
//     flag = !flag;
// }

// // given a window of size 2, we want to find the biggest drops 
// // so an O(n) approach is simply just iterating through and calculating the difference between subsequent elements 

// let mut white_probs_diffs: Vec<f64> = Vec::new();
// let mut priority_queue = BinaryHeap::new();

// for i in (2..final_probabilities.len()).step_by(2) {
//     // (((1.0 - final_probabilities[i]) * 100.0).round() / 100.0);
//     priority_queue.push(MovesTuple(((final_probabilities[i - 2] - final_probabilities[i]) * 100.0).round() / 100.0, i as i32));
//     // white_probs_diffs[i] = final_probabilities[i] - final_probabilities[i - 2]; 
// }

// // poll the 3 top elements 
// // sort by index because that will be what's required for our feedback

// let mut count = 3;
// let mut biggest_diffs: Vec<(f64, i32)> = Vec::new();

// while count != 0 {

//     match priority_queue.pop() {
//         Some(curr_touple) => {
//             biggest_diffs.push((curr_touple.0, curr_touple.1));
//             count = count - 1;
//         }
//         None => {
//             println!("None value at pq");
//             count = count - 1;
//         }
//     }
// }

// let final_best_moves =  best_moves.lock().unwrap();
// biggest_diffs.sort_by(|a,b| a.1.cmp(&b.1));

// for i in 0..biggest_diffs.len() {
//     println!("BIGGEST DIFF FOUND = {} AT ROUND = {} BEST MOVE = {}, PLAYED MOVE = {}", biggest_diffs[i].0, biggest_diffs[i].1 + 1, final_best_moves[biggest_diffs[i].1 as usize], lan_moves[biggest_diffs[i].1 as usize]);
// }
// let duration = start.elapsed();

// println!("Program time: {}", duration.as_millis());

// // parsed_game_data.best_moves = best_moves.lock().unwrap().clone();
// parsed_game_data.fen_values = fen_values;
// parsed_game_data.lan_moves = lan_moves;
// // parsed_game_data.probabilities = probabilities.lock().unwrap().clone();

// match chess_proto_response_thread.lock() {
//     Ok(mut guard) => {
//         guard.games.push(parsed_game_data);
//         println!("chess games len = {}", guard.games.len());
//     }
//     Err(poison_error) => {
//         println!("poisoned mutex");
//     }
// }


// println!("yo");

#[tonic::async_trait]
impl Greeter for MyGreeter {
    async fn say_hello(
        &self,
        request: Request<HelloRequest>,
    ) -> Result<Response<HelloReply>, Status> {
        println!("Got a request: {:?}", request);

        let reply = HelloReply {
            message: format!("Hello {}!", request.into_inner().name)
        };

        Ok(Response::new(reply))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let addr = "[::1]:50051".parse()?;
    // let greeter = MyGreeter::default();

    let origin_strings = vec![
        "http://localhost:3000",
        // Add more allowed origins here if needed
    ];
    
    // Convert &str to tonic::codegen::http::HeaderValue
    let allowed_origins: Vec<HeaderValue> = origin_strings
        .into_iter()
        .map(|origin| {
            HeaderValue::from_str(origin).expect("Invalid origin string")
        })
        .collect();
    
    // Configure CORS to allow your React app's origin
    let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods(Any)
    .allow_headers(Any);

    let chess_struct = ChessStruct::new();

    let chess_service = ChessServiceServer::new(chess_struct);

    let server = Server::builder()
    .layer(GrpcWebLayer::new())
    .layer(cors)
    .add_service(chess_service)
    .serve(([0,0,0,0], 50051).into())
    .await?;

    Ok(())
}

// api link

// "https://api.chess.com/pub/player/noopdogg07/games/2024/010
