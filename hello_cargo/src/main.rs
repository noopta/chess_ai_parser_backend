use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn main() -> std::io::Result<()> {
    let mut child = Command::new("../Stockfish/src/stockfish")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;

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