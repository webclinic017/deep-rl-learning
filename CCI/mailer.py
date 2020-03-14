import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from requests.auth import HTTPBasicAuth


class SendMail:
    def __init__(self):
        self.from_address = "thinhle.ai@gmail.com"
        self.password = "kunegiucpavhdutu"

    def notification(self, txt):
        msg = MIMEMultipart()
        msg['Subject'] = "Close notification"
        msg['Form'] = 'thinhlx1993@gmail.com'
        html = """
            <html>
                <body>
                    <h1>Summary</h1>
                    
                </body>
            
            </html>
        
        """.format(txt)
        part1 = MIMEText(html, 'html')
        msg.attach(part1)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.from_address, self.password)
            server.sendmail(
                self.from_address,
                "thinhlx1993@gmail.com",
                msg.as_string()
            )


if __name__ == '__main__':
    mailer = SendMail()
    mailer.notification("hell0")
