import argparse
import sys
from src.model import QueryModel
import logging
from src.logging import setup_logging

if __name__ == "__main__":
    setup_logging(logging.INFO)

    try:
        model_path = "models/trained.model"

        logging.info(f"Loading trained model from: {model_path}")

        query_model = QueryModel("models/trained.model", "legit")

        logging.debug("Parsing arguments")

        ap = argparse.ArgumentParser()
        ap.add_argument("domain", nargs = '*', help="Domain(s) to be test. Either single domain or comma separated")
        ap.add_argument("-i", "--interactive", action='store_true',
                        help="Enter interactive mode to type in many domains")

        args = vars(ap.parse_args())

        logging.debug(f"Arguments parsed: {args}")

        given_domain = args["domain"]
        result = None

        if args["interactive"]:

            first_prediction = None
            if given_domain:
                logging.debug(f"Domain was given in arguments: {given_domain}")

                first_prediction = query_model.predict(given_domain)

                logging.debug(f"first_prediction: {first_prediction}")

            logging.debug("Entering interactive mode")

            result = query_model.interactive()

            logging.info("Leaving interactive mode")

            if given_domain and first_prediction is not None:
                result = result and first_prediction

                logging.debug(f"result: {result}")
        else:
            if not given_domain:
                message = "Must specify a domain(s) to be tested"
                logging.error(message)
                sys.exit(2)

            result = query_model.predict(given_domain)

            logging.debug(f"result: {result}")

        if result is None:
            message = "No predictions were made"
            logging.warning(message)
            sys.exit(2)

        if not result:
            message = "Non-Legit Domain(s) Detected"
            logging.warning("Non-Legit Domain(s) Detected")
            sys.exit(3)

        logging.info("Legit Domain(s)")
        sys.exit(0)
    except Exception:
        message = "An Uncaught Exception Occurred"
        logging.exception(message)
