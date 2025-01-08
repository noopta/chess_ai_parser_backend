use tonic::{transport::Server, Request, Response, Status};
use reqwest::header;
use reqwest::Client;

// use hello_world::greeter_server::{Greeter, GreeterServer};
use hello_world::greeter_server::{Greeter};
use hello_world::{HelloReply, HelloRequest};
use hello_world::chess_service_server::{ChessService, ChessServiceServer};
use hello_world::{ProfileRequestData, ChessApiGameResponse, ChessPlayer, Game};

pub mod hello_world {
    tonic::include_proto!("hello_cargo");
}

#[derive(Debug, Default)]
pub struct MyGreeter {}

#[derive(Debug, Default)]
pub struct ChessStruct{}

#[derive(serde::Deserialize, Debug)]
struct LocalChessApiGameResponse {
    games: Vec<LocalGame>
}

#[derive(serde::Deserialize, Debug)]
struct LocalGame {
    url: String,
    pgn: String,
    tcn: String,
    fen: String,
    white: LocalChessPlayer,
    black: LocalChessPlayer
}

#[derive(serde::Deserialize, Debug)]
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

// fn convert_game(current_game: ChessApiGameResponse) -> hello_world::Game {
//     hello_world::Game {
//         url: current_game.url,
//         pgn: current_game.pgn,
//         tcn: current_game.tcn,
//         fen: current_game.fen,
//         white: Some(hello_world::Player {
//             rating: current_game.white.rating,
//             result: current_game.white.result,
//             username: current_game.white.username
//         }),
//         black: Some(hello_world::Player {
//             rating: current_game.black.rating,
//             result: current_game.black.result,
//             username: current_game.black.username
//         }),
//     }
// }

pub async fn fetch_games(username: &str, month: &str, year: &str) -> Result<LocalChessApiGameResponse, Box<dyn std::error::Error>> {
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

    println!("ENDPOINT: {}", _endpoint_url);

    // let response_text = reqwest::get(&_endpoint_url).await?;

    // let chess_games: Vec<ChessApiGameResponse> = serde_json::from_str(&response_text)?;
    // let parsed_games: Vec<Game> = chess_games.into_iter().map(convert_game).collect();
    
    if response.status().is_success() {
        let json_response: LocalChessApiGameResponse = response.json().await?;
        println!("RAW JSON: {:?}", json_response);
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
                println!("Fetched chess response: {:?}", chess_response); // Print the raw fetched data

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