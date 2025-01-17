use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};


use tokio::io::split;
use tonic::{transport::Server, Request, Response, Status};
use reqwest::header;
use reqwest::Client;
use std::sync::Arc;
use std::sync::Mutex;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// use hello_world::greeter_server::{Greeter, GreeterServer};
use hello_world::greeter_server::{Greeter};
use hello_world::{HelloReply, HelloRequest};
use hello_world::chess_service_server::{ChessService, ChessServiceServer};
use hello_world::{ProfileRequestData};
use regex::Regex;

use chess::{Board, ChessMove, Piece, Square};
use std::str::FromStr;
use std::time::Instant;

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

    let exponent: f64 = -1.0 * (float_cp / 400.0);

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


pub mod hello_world {
    tonic::include_proto!("hello_cargo");
}

#[derive(Debug, Default)]
pub struct MyGreeter {}

#[derive(Debug, Default)]
pub struct ChessStruct{}

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
    fn into_proto(self) -> hello_world::ChessApiGameResponse {
        hello_world::ChessApiGameResponse {
            // map local games into proto games
            games: self.games.into_iter().map(|g| g.into_proto()).collect(),
        }
    }
}

impl LocalGame {
    fn into_proto(self) -> hello_world::Game {
        hello_world::Game {
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
    fn into_proto(self) -> hello_world::ChessPlayer {
        hello_world::ChessPlayer {
            rating: self.rating,
            result: self.result,
            username: self.username,
        }
    }
}



async fn fetch_games(username: &str, month: &str, year: &str) -> Result<LocalChessApiGameResponse, Box<dyn std::error::Error>> {
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

#[tonic::async_trait]
impl ChessService for ChessStruct {
    async fn get_chess_games(
        &self,
        request: Request<ProfileRequestData>,
    ) -> Result<Response<hello_world::ChessApiGameResponse >, Status> {
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

        match fetch_games(username, month, year).await {
            Ok(chess_response) => {
                // println!("Fetched chess response: {:?}", chess_response); // Print the raw fetched data
                
                // let request_ref = request.get_ref().clone();


                // so the way stockfish works is that it takes in a FEN and recommends the next best move from there 
                // so.. one way is to get every individual fen in the game 

                // is there another way?

                let games = chess_response.games.clone();

                let first_game = games.get(0).unwrap();

                let chess_san = convert_to_san(&first_game.pgn);

                let split_moves = chess_san
                    .split(", ")
                    .map(|s| s.to_string())
                    .collect();

                let lan_moves: Vec<String> = convert_pgn_moves_to_lan(split_moves);
                let lan_moves_copy = lan_moves.clone();

                // let's just focus on creating the functions necessary for one game, 
                // then ideally we should have the approach for all games at scale

                // here we have a string of san 
                // we can get the probabilities of both black and white at each point
                let split_san = chess_san.split(", ");
                
                let mut probabilites: Vec<f64> = Vec::new();

                let mut child = Command::new("../Stockfish/src/stockfish")
                // creates a standard input pipe
                    .stdin(Stdio::piped())
                    // creates a standard output pipe 
                    .stdout(Stdio::piped())
                    .spawn()?; // spawns the child process
            
                let mut stdin = child.stdin.take().unwrap();
                let stdout = child.stdout.take().unwrap();
                let mut depth_one_cp: f64;
                let mut probabilities: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
                let probabilities_thread: Arc<Mutex<Vec<f64>>> = Arc::clone(&probabilities);

                let mut best_moves: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
                let mut best_moves_thread: Arc<Mutex<Vec<String>>> = Arc::clone(&best_moves);

                let mut fen_values: Vec<String> = Vec::new();
            
                let reader_thread = std::thread::spawn(move || {
                    let reader = BufReader::new(stdout);
            
                    for line in reader.lines() {
                        if let Ok(line) = line {
                            if line.starts_with("info depth 1") {
                                // println!("ENGINE: {}", line);
                                let split_line: Vec<&str> = line.split(" ").collect();

                                for i in 0..split_line.len() {
                                    if split_line[i].to_string().starts_with("cp") {
                                        let parsed_value: i32 = split_line[i + 1].parse().expect("Not a valid integer");
                                        let mut probs = probabilities_thread.lock().unwrap();
                                      
                                        // probabilities_thread.push(convert_cp_to_chances(&parsed_value));
                                        // (x * 100.0).round() / 100.0
                                        probs.push((convert_cp_to_chances(&parsed_value) * 100.0).round() / 100.0);
                                        // probs.push(convert_cp_to_chances(&parsed_value));
                                        break;
                                    } 
                                }
//                                 player: black || move: a8h1 || prob = 0.02177404821346721
// player: white || move: h3g4 || prob = 0.9908586214747741

                            } else if line.starts_with("bestmove") {
                                // println!("ENGINE BEST MOVE: {}", line);
                                let mut moves = best_moves_thread.lock().unwrap();

                                moves.push(line);
                            }
                        }
                    }
                });

                let mut current_moves: String = "".to_string();
                let mut count = 3;

                // writeln!(stdin, "position startpos")?;
                // writeln!(stdin, "go depth 10")?;
                writeln!(stdin, "uci")?;
                writeln!(stdin, "isready")?;
                for curr_move in &lan_moves {
                    // feed move by move to stockfish and store the result 
                    // in an array

                    // add the current move to the current_moves
                    // then feed the current_moves string to stockfish
                    // tell stockfish to go a certain depth 

                    // then read the lines back 
                    
                    current_moves.push_str(&curr_move);
                    
                    fen_values.push(convert_lan_to_fen(&current_moves));
                    println!("FEN: {}", convert_lan_to_fen(&current_moves));
                    println!("current move: {}", curr_move);
                    current_moves.push_str(" ");
                    // writeln!(stdin, "position startpos moves {}", current_moves)?;
     
                    writeln!(stdin, "position startpos moves {}", current_moves)?;
                    writeln!(stdin, "go depth 3")?;
                }

                writeln!(stdin, "quit")?;


                // println!("game in SAN: {}", convert_to_san(&first_game.pgn));
                let cp: i32 = 100;

                let result = convert_cp_to_chances(&cp);

                // println!("result cp: {}", result.to_string());

                let proto_response = chess_response.into_proto();
                reader_thread.join().expect("Issue closing reader thread");
                
                let final_probabilities =  probabilities.lock().unwrap();

                // for curr_prob in final_probabilities.iter() {
                //     println!("prob = {}", curr_prob);
                // }
                let mut player: String = "white".to_string();
                let mut white_probs: Vec<f64> = Vec::new();
                let mut black_probs: Vec<f64> = Vec::new();


                for i in 0..final_probabilities.len() {
                    black_probs.push(((1.0 - final_probabilities[i]) * 100.0).round() / 100.0);
                }

                // so this won't work if the other probability + curr > 1 or less than 1
                // so just pick one

                // each probabilty calculated is white's chance to win
                let mut flag = false;

                for i in 0..black_probs.len() {
                    if(!flag) {
                        println!("white = {} black = {} move = {}", final_probabilities[i], black_probs[i], lan_moves[i]);
                    }
                    
                    flag = !flag;
                }

                // given a window of size 2, we want to find the biggest drops 
                // so an O(n) approach is simply just iterating through and calculating the difference between subsequent elements 

                let mut white_probs_diffs: Vec<f64> = Vec::new();
                let mut priority_queue = BinaryHeap::new();

                for i in (2..final_probabilities.len()).step_by(2) {
                    // (((1.0 - final_probabilities[i]) * 100.0).round() / 100.0);
                    priority_queue.push(MovesTuple(((final_probabilities[i - 2] - final_probabilities[i]) * 100.0).round() / 100.0, i as i32));
                    // white_probs_diffs[i] = final_probabilities[i] - final_probabilities[i - 2]; 
                }

                // poll the 3 top elements 
                // sort by index because that will be what's required for our feedback

                let mut count = 3;
                let mut biggest_diffs: Vec<(f64, i32)> = Vec::new();

                while count != 0 {
                    let curr_touple = priority_queue.pop().unwrap();
                    biggest_diffs.push((curr_touple.0, curr_touple.1));
                    count = count - 1;
                }

                let final_best_moves =  best_moves.lock().unwrap();
                biggest_diffs.sort_by(|a,b| a.1.cmp(&b.1));

                for i in 0..biggest_diffs.len() {
                    println!("BIGGEST DIFF FOUND = {} AT ROUND = {} BEST MOVE = {}, PLAYED MOVE = {}", biggest_diffs[i].0, biggest_diffs[i].1 + 1, final_best_moves[biggest_diffs[i].1 as usize], lan_moves[biggest_diffs[i].1 as usize]);
                }
                let duration = start.elapsed();

                println!("Program time: {}", duration.as_millis());
                Ok(Response::new(proto_response))
            },
            Err(e) => Err(Status::internal(format!("Error fetching games: {}", e))),
        }
        // https://api.chess.com/pub/player/noopdogg07/games/2024/010
        // Ok(Response::new(reply))
    }
}

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
    let addr = "[::1]:50051".parse()?;
    // let greeter = MyGreeter::default();

    let chess_service = ChessStruct::default();

    // Server::builder()
    //     .add_service(GreeterServer::new(greeter))
    //     .serve(addr)
    //     .await?;


    Server::builder()
    .add_service(ChessServiceServer::new(chess_service))
    .serve(addr)
    .await?;

    Ok(())
}

// api link

// "https://api.chess.com/pub/player/noopdogg07/games/2024/010
