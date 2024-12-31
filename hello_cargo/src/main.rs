use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn main() -> std::io::Result<()> {
    
    get_chess_data();
    // mutable variable that creates a new child process based on a binary 
    // Command runs a CLI based command essentially 
    let mut child = Command::new("../Stockfish/src/stockfish")
    // creates a standard input pipe
        .stdin(Stdio::piped())
        // creates a standard output pipe 
        .stdout(Stdio::piped())
        .spawn()?; // spawns the child process

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();

    let reader_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            if let Ok(line) = line {
                println!("ENGINE: {}", line);
            }
        }
    });

    writeln!(stdin, "uci")?;
    writeln!(stdin, "isready")?;
    writeln!(stdin, "position startpos")?;
    writeln!(stdin, "go depth 10")?;

    reader_thread.join().unwrap();
    Ok(())
    
}


fn get_chess_data() {
    println!("hi")
}