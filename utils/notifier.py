# utils/notifier.py

import smtplib
from email.mime.text import MIMEText
import logging

def send_email_notification(subject, body, config):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = config['notifier']['email_sender']
        msg['To'] = config['notifier']['email_receiver']

        with smtplib.SMTP_SSL(config['notifier']['smtp_server'], config['notifier']['smtp_port']) as server:
            server.login(config['notifier']['email_sender'], config['notifier']['email_password'])
            server.send_message(msg)

        logging.info("[NOTIFIER] Email notification sent.")

    except Exception as e:
        logging.error(f"[NOTIFIER] Failed to send email: {e}")
