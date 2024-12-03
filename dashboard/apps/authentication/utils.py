import os
import smtplib
import ssl
import requests
from django.conf import settings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(
        username: str,
        receiver_email: str,
        email: str

):

    sender_email = os.environ["email_account"]
    message = MIMEMultipart()
    message["Subject"] = "New User Registration to MAIA"
    message["From"] = sender_email
    message["To"] = receiver_email

    html = """\
    <html>
        <head></head>
        <body>
            <p>A New user has requested to join MAIA.</p>
            <p>Username: {}</p>
            <p>E-mail: {}</p>

        </body>
    </html>
    """.format(username, email)

    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(html, "html")

    message.attach(part1)

    port = 465  # For SSL
    password = os.environ["email_password"]

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())


def send_discord_message(username, namespace):
    data = {
        "content": f"{username} is requesting a MAIA account for the project {namespace}.",
        "username": "MAIA-Bot"
    }

    # leave this out if you dont want an embed
    # for all params, see https://discordapp.com/developers/docs/resources/channel#embed-object
    data["embeds"] = [
        {
            "description": "MAIA User Registration Request",
            "title": "MAIA Account Request",
        }
    ]
    url = settings.DISCORD_URL

    result = requests.post(url, json=data)

    try:
        result.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
    else:
        print("Payload delivered successfully, code {}.".format(result.status_code))