import os
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from MAIA.dashboard_utils import (
    send_maia_message_email,
)
from MAIA.keycloak_utils import get_maia_users_from_keycloak

# load env variables
from dotenv import load_dotenv

load_dotenv()
import json


class Settings:
    def __init__(self, settings_dict):
        for key, value in settings_dict.items():
            setattr(self, key, value)


def send_all_users_reminder_email(settings_dict, email_list=None):
    """
    Sends a reminder email to all MAIA users using the content from reminder_email.md.

    Parameters
    ----------
    settings_dict : dict
        The dictionary containing configuration values.

    Returns
    -------
    tuple
        (int, list): Number of emails sent and list of any failed email addresses
    """
    # Convert dictionary to Settings object
    settings = Settings(settings_dict)
    if email_list is None:
        maia_users = get_maia_users_from_keycloak(settings)

        # Get list of all user emails
        user_emails = [user["email"] for user in maia_users]
    else:
        user_emails = email_list

    # Required environment variables for sending emails
    required_env_vars = ["email_account", "email_password", "email_smtp_server"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Read the email template
    email_template_path = Path(__file__).parent / "reminder_email.html"
    try:
        with open(email_template_path, "r") as f:
            email_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Email template not found at {email_template_path}")
    # Send the email to all users
    success = send_maia_message_email(receiver_emails=user_emails, subject="MAIA Platform Updates", message_body=email_content)

    if success:
        print("Email sent successfully")
        return len(user_emails), []
    else:
        print("Email failed to send")
        return 0, user_emails


if __name__ == "__main__":
    # This allows the script to be run directly for testing
    # load settings
    with open("settings.json") as f:
        settings_dict = json.load(f)

    email_list = ["xxx@live.com"]
    num_sent, failed = send_all_users_reminder_email(settings_dict, email_list)
    print(f"Successfully sent {num_sent} emails")
    if failed:
        print(f"Failed to send emails to: {', '.join(failed)}")
