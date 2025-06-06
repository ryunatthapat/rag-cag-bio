import sys
from colorama import Fore, Style, init as colorama_init
from rag import rag_answer
from cag import cag_answer

colorama_init(autoreset=True)

def print_result(module, answer, timing, color):
    print(f"{color}[{module}] {Style.RESET_ALL}({timing:.2f}s): {answer}")

def main():
    print(f"{Fore.CYAN}RAG & CAG Biographies Demo CLI{Style.RESET_ALL}")
    print("Type your question (or 'exit' to quit):\n")
    while True:
        try:
            query = input(f"{Fore.YELLOW}> {Style.RESET_ALL}").strip()
            if query.lower() in {"q", "exit", "quit"}:
                print("Goodbye!")
                break
            if not query:
                continue
            print(f"\n{Fore.MAGENTA}Processing...{Style.RESET_ALL}")
            # RAG
            rag_ans, rag_timing = rag_answer(query)
            # CAG
            cag_ans, cag_timing = cag_answer(query)
            print()
            print_result("RAG", rag_ans, rag_timing.get("retrieval", 0) + rag_timing.get("generation", 0), Fore.GREEN)
            print_result("CAG", cag_ans, cag_timing.get("generation", 0), Fore.BLUE)
            print()
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"{Fore.RED}[Error]{Style.RESET_ALL} {e}\n")

if __name__ == "__main__":
    main() 