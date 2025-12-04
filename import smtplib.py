import smtplib

server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
server.login("limor@nurexone.com", "jnzihghuoquiaqvq")
print("Login successful ✅")
server.quit()