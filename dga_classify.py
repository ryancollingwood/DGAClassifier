import argparse


def test_domain(domain: str):
    pass


def interactive():
    print("enter a domain (e.g. google) or type !q to quit")
    input_value = ""
    while input_value != "!q":
        input_value = input(">").strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", type=str, help="Domain to be tested")
    ap.add_argument("-i", "--interactive", required=False, default=False,
                    help="Enter interactive mode to type in many domains")

    args = vars(ap.parse_args())

    given_domain = args["domain"].strip()

    if "interactive" in args:
        if given_domain != "":
            test_domain(given_domain)
        interactive()
    else:
        test_domain(given_domain)
