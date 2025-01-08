// use hello_world::greeter_client::GreeterClient;
// use hello_world::{HelloRequest, ProfileRequestData};
use hello_world::{ProfileRequestData};
use hello_world::chess_service_client::ChessServiceClient;

pub mod hello_world {
    tonic::include_proto!("hello_cargo");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let mut client = GreeterClient::connect("http://[::1]:50051").await?;

    // let request = tonic::Request::new(HelloRequest {
    //     name: "Tonic".into(),
    // });

    // let response = client.say_hello(request).await?;

    // println!("RESPONSE={:?}", response.into_inner().message);


    let mut client = ChessServiceClient::connect("http://[::1]:50051").await?;

    let request = tonic::Request::new(ProfileRequestData {
        username: "noopdogg07".into(),
        month: "10".into(),
        year: "2024".into(),
        number_of_games: "2".into()
    });

    let response = client.get_chess_games(request).await?;
    
    println!("RESPONSE={:?}", response);

    Ok(())
}


// async fn test() -> Result<(), Box<dyn std::error::Error>> {
//     let mut client = ChessServiceClient::connect("http://[::1]:50051").await?;

//     let request = tonic::Request::new(ProfileRequestData {
//         username: "noopdogg07".into(),
//         month: "10".into(),
//         year: "2024".into(),
//         number_of_games: "2".into()
//     });

//     let response = client.get_chess_games(request).await?;
    
//     println!("RESPONSE={:?}", response);

//     Ok(())
// }