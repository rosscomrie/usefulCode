import csv

class CSVParser:
    """
    A utility class for parsing CSV files.
    """

    @staticmethod
    def loadCSV(filepath):
        """
        Loads a CSV file and converts it into a list of dictionaries.

        Args:
            filepath (str): The path to the CSV file.

        Returns:
            list: A list of dictionaries representing the CSV data.
        """
        with open(filepath, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]
        return data
        
    @staticmethod
    def filterRows(data, columnName, columnValue):
        """
        Filters rows in the CSV data based on a column's value.

        Args:
            data (list): The list of dictionaries representing the CSV data.
            columnName (str): The name of the column to filter on.
            columnValue (str): The value to filter for in the specified column.

        Returns:
            list: A list of dictionaries representing the filtered CSV data.
        """
        filtered_data = [row for row in data if row[columnName] == columnValue]
        return filtered_data