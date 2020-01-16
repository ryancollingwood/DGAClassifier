import logging
import pandas as pd
from .load_model import load_model
from src.logging import setup_logging


def generate_domain(year: int, month: int, day: int) -> str:
    """
    Generate a domain based on a date, adapted from:
    Evolving Smart URL Filter in a Zone-Based Policy Firewall for Detecting Algorithmically Generated Malicious Domains
    Gammerman, A., Vovk, V. and Papadopoulos, H., 2015. Statistical Learning and Data Sciences.

    https://www.researchgate.net/profile/Konstantinos_Demertzis/publication/333797659_Evolving_Smart_URL_Filter_in_a_Zone-Based_Policy_Firewall_for_Detecting_Algorithmically_Generated_Malicious_Domains/links/5d0bc83ca6fdcc246297b318/Evolving-Smart-URL-Filter-in-a-Zone-Based-Policy-Firewall-for-Detecting-Algorithmically-Generated-Malicious-Domains.pdf

    :param year: int
    :param month: int
    :param day: int
    :return: str
    """
    domain = ""

    for i in range(16):
        year = ((year ^ 8 * year) >> 11) ^ ((year & 0xFFFFFFF0) << 17)
        month = ((month ^ 4 * month) >> 25) ^ 16 * (month & 0xFFFFFFF8)
        day = ((day ^ (day << 13)) >> 19) ^ ((day & 0xFFFFFFFE) << 12)
        domain += chr(((year ^ month ^ day) % 25) + 97)

    logging.debug(f"Generated Domain: {domain}")

    return domain


def test_model(filename = "models/trained.model"):
    """
    Load the model specified and pass in some tricky examples

    :param filename:
    :return:
    """
    setup_logging(logging.INFO)

    logging.debug("test_model")
    logging.debug(f"filename: {filename}")

    loaded_model = load_model(filename)

    blind_test = [
        "google",
        "asx",
        "netflix",
        "stan",
        "youtube",
        "facebook",
        "bing",
        "duckduckgo",
        "kjhkhssf",
        "scikit-learn",
        generate_domain(1983, 7, 1),
        "reddit",
        "longestdomains",
        "zwpejkljhdpoqk",
        "pklwllpppqzibn",
        "stackoverflow",
        generate_domain(2019, 12, 17),
        generate_domain(1900, 3, 22),
    ]

    # create some permutations of date a long, long, time ago in a galaxy far, far, way
    test_gen = generate_domain(1983, 7, 1)
    for i in range(1, 6):
        val = test_gen[::i]
        if val:
            blind_test.append(test_gen[::i])
        for j in range(1, 6):
            val = test_gen[j:i]
            if val:
                blind_test.append(test_gen[j:i])

    logging.debug(blind_test)

    test_df = pd.DataFrame(blind_test, columns=["domain"])

    logging.debug("About to make predictions on test_df")

    results = loaded_model.predict(test_df.values)

    for i, result in enumerate(results):
        logging.info(f"{blind_test[i]}: {result}")


if __name__ == "__main__":
    test_model()
