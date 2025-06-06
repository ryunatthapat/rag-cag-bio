import sys
from colorama import Fore, Style, init as colorama_init
from rag import rag_answer
from cag import cag_answer
import threading

colorama_init(autoreset=True)

def print_result(module, answer, timing, color):
    print(f"{color}[{module}] {Style.RESET_ALL}({timing:.2f}s): {answer}")

def main():
    print(f"{Fore.CYAN}RAG & CAG Company FAQ Demo CLI{Style.RESET_ALL}")
    print("Type your question about PALO IT (or 'exit' to quit):\n")
    while True:
        try:
            query = input(f"{Fore.YELLOW}> {Style.RESET_ALL}").strip()
            if query.lower() in {"q", "exit", "quit"}:
                print("Goodbye!")
                break
            if not query:
                continue
            print(f"\n{Fore.MAGENTA}Processing...{Style.RESET_ALL}")
            results = {}
            exceptions = {}
            def run_rag():
                try:
                    rag_ans, rag_timing = rag_answer(query)
                    results['rag'] = (rag_ans, rag_timing)
                    print_result("RAG", rag_ans, rag_timing.get("retrieval", 0) + rag_timing.get("generation", 0), Fore.GREEN)
                except Exception as e:
                    exceptions['rag'] = e
                    print(f"{Fore.RED}[RAG Error]{Style.RESET_ALL} {e}")
            def run_cag():
                try:
                    cag_ans, cag_timing = cag_answer(query)
                    results['cag'] = (cag_ans, cag_timing)
                    print_result("CAG", cag_ans, cag_timing.get("generation", 0), Fore.BLUE)
                except Exception as e:
                    exceptions['cag'] = e
                    print(f"{Fore.RED}[CAG Error]{Style.RESET_ALL} {e}")
            t_rag = threading.Thread(target=run_rag)
            t_cag = threading.Thread(target=run_cag)
            t_rag.start()
            t_cag.start()
            t_rag.join()
            t_cag.join()
            print()
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break
        except Exception as e:
            print(f"{Fore.RED}[Error]{Style.RESET_ALL} {e}\n")

if __name__ == "__main__":
    main() 