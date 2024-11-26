from argparse import ArgumentParser

import requests
from urllib.parse import urlparse
from pathlib import Path

# add utils package to path
import sys
sys.path.insert(0, (Path(__file__).parent.parent).resolve().as_posix())

from utils import setup_logging


def strip_path(url):
    if not url.startswith('http'):
        url = 'https://' + url
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--address", type=str, help="URL to Label Studio (e.g. https://my-labelstudio.mycloud.com)", required=True)
    parser.add_argument("--token", type=str, help="API token of Label Studio", required=True)
    parser.add_argument("--user-to-delete", type=str, help="E-mail address of user to delete", required=True)

    parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()
    # setup logger
    logger = setup_logging(__file__, level=args.logging_level)
    logger.debug(f"Input arguments: {args}")

    address = strip_path(args.address)

    user_mail = args.user_to_delete

    kwargs = {
        "headers": {"Authorization": f"Token {args.token}"},
        "verify": False,
    }

    # get all users
    logger.info(f"Looking for user {user_mail} at {address}")
    response = requests.request(
        "GET",
        f"{address}/api/users/",
        **kwargs
    )
    users = response.json()

    user_id = None
    for usr in users:
        if usr["email"] == user_mail:
            user_id = usr["id"]
            break
    if user_id is None:
        raise Exception(f"User {user_mail} does not exist.")

    logger.info(f"Deleting user {user_id} ({user_mail}) from {address}")
    response = requests.request("DELETE", f"{address}/api/users/{user_id}/", **kwargs)
