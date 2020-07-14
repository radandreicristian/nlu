from typing import Any

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from util.result.abstract_result_writer import AbstractResultWriter


class CloudResultWriter(AbstractResultWriter):

    def __init__(self, secret_path: str, spreadsheet_name: str):
        self.secret_path = secret_path
        self.spreadsheet_name = spreadsheet_name

    def get_destination(self) -> Any:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name('thesis_secret.json', scope)
        client = gspread.authorize(credentials)
        return client.open(self.spreadsheet_name).sheet1

    def append_row(self, values: list, index: int) -> None:
        worksheet = self.get_destination()
        worksheet.insert_row(values, index)

    def save(self) -> None:
        # Well, it's saved on google docs, so, we don't need to save it.
        pass
