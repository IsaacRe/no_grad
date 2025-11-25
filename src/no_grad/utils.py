class Logger:

    def __init__(self, *metrics, column_width: int = 12):
        self.metrics, self.fmt_strings = zip(*metrics)
        self.column_width = column_width

    def print_header(self, end: str = "\n"):
        header = ""
        for metric in self.metrics:
            metric = metric[:self.column_width]
            total_padding = self.column_width - len(metric)
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            header += " " * left_padding + metric + " " * right_padding + "|"
        print("-" * len(header))
        print(header[:-1], end=end)

    def print_metrics(self, *values, end: str = "\n"):
        row = ""
        for value, fmt_string in zip(values, self.fmt_strings):
            value_str = fmt_string.format(value)[:self.column_width]
            total_padding = self.column_width - len(value_str)
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            row += " " * left_padding + value_str + " " * right_padding + "|"
        print(row[:-1], end=end)
