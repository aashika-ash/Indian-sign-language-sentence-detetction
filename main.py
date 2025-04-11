import subprocess
import time

def run_servers():
    # Start the letter recognition server
    letter_process = subprocess.Popen(["python", r"C:\Users\dell\islbridge\backend\app_letter.py"])
    print("Letter recognition server started on port 5000")
    
    # Start the sentence recognition server
    sentence_process = subprocess.Popen(["python", r"C:\Users\dell\islbridge\backend\app_sentence.py"])
    print("Sentence recognition server started on port 5001")
    
    return letter_process, sentence_process

def main():
    letter_process, sentence_process = run_servers()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        letter_process.terminate()
        sentence_process.terminate()
        print("Servers stopped successfully")

if __name__ == '__main__':
    main()