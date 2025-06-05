from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os.path
import pickle
from tqdm import tqdm

# Authenticate and initialize the API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
creds = None

# Load or create new credentials
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
else:
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('drive', 'v3', credentials=creds)

# List of folder IDs and their corresponding names
folders = [
    {'id': '1AwIpw-AV-wovfEyFp_RQSmBaiVUd4Dpc', 'name': 'prompt-15 (type 1)'},
    {'id': '1WPGJbD0znzcSREtEjQ4jy5dfFDrjfh5l', 'name': 'prompt-17 (type 3)'},
    {'id': '19WnlFdpMttHR_GaYvjbbkuXcYljFmIrs', 'name': 'prompt-18 (type 4)'},
]

# Open a file to save the comments
with open('comments.txt', 'w') as output_file:
    for folder in tqdm(folders, desc="Processing folders"):
        folder_id = folder['id']
        folder_name = folder['name']
        # Get all files in the specified folder
        results = service.files().list(q=f"'{folder_id}' in parents and mimeType contains 'image'").execute()
        files = results.get('files', [])

        # Extract comments
        for file in files:
            file_id = file['id']
            file_name = file['name']
            comments = service.comments().list(fileId=file_id, fields="comments(content)").execute()
            for comment in comments.get('comments', []):
                comment_content = comment['content']
                # Split comments based on "-"
                split_comments = comment_content.split('- ')
                for split_comment in split_comments:
                    if split_comment.strip():  # Check if the comment is not empty
                        output_file.write(f"Folder: {folder_name}|||File: {file_name}|||Comment: {split_comment.strip()}\n")