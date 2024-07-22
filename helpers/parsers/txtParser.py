import os

class TXTParser:
    """
    A class for parsing TXT files.
    """

    @staticmethod
    def loadTXT(filepath):
        """
        Load the content of a TXT file.

        Args:
            filepath (str): The path to the TXT file.

        Returns:
            str: The content of the TXT file.
        """
        with open(filepath, 'r', encoding="utf8") as file:
            content = file.read()
        return content