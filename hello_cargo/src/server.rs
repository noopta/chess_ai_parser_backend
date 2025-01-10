use tokio::io::split;
use tonic::{transport::Server, Request, Response, Status};
use reqwest::header;
use reqwest::Client;

// use hello_world::greeter_server::{Greeter, GreeterServer};
use hello_world::greeter_server::{Greeter};
use hello_world::{HelloReply, HelloRequest};
use hello_world::chess_service_server::{ChessService, ChessServiceServer};
use hello_world::{ProfileRequestData};
use regex::Regex;

fn parse_game_pgn(pgn: &str) -> String {

    let split_pgn: Vec<&str> = pgn.split("\n").collect();

    let game_moves = split_pgn[split_pgn.len() - 2];

    // println!("game move: {}", game_moves.to_string());
    // for line in split_pgn {
    //     println!("Line: {}", line.to_string());
    // }

    "abc".to_string()
}  

fn convert_cp_to_chances(cp: &i32) -> f64{
    let float_cp: f64 = *cp as f64;

    let exponent: f64 = -1.0 * (float_cp / 400.0);

    println!("exponent = {}", exponent.to_string());

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
    let final_output = output_moves.join(", ");
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

                // let's just focus on creating the functions necessary for one game, 
                // then ideally we should have the approach for all games at scale

                // here we have a string of san 
                // we can get the probabilities of both black and white at each point
                let split_san = chess_san.split(", ");
                
                let mut probabilites: Vec<f64> = Vec::new();
                
                for curr_move in split_san {
                    // feed move by move to stockfish and store the result 
                    // in an array
                    println!("move: {}", curr_move);
                }

                // println!("game in SAN: {}", convert_to_san(&first_game.pgn));
                let cp: i32 = 100;

                let result = convert_cp_to_chances(&cp);

                println!("result cp: {}", result.to_string());

                let proto_response = chess_response.into_proto();
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