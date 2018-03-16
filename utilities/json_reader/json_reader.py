from flask import Flask, request, make_response, redirect, url_for
import sys
import gzip
app = Flask(__name__)

@app.route("/")
def default():
    kwargs = {'from': 0, 'n': 30}
    return redirect(url_for('explorer', **kwargs))

@app.route("/show")
def explorer():
    first_row = int(request.args.get('from', 0))
    num_rows = int(request.args.get('n', 30))
    last_row = int(request.args.get('to', first_row + num_rows))

    if app.config.get('json_file').endswith('.gz'):
        open_func = gzip.open
    else:
        open_func = open

    with open_func(app.config.get('json_file'), 'rt', encoding='utf8') as fp:
        lines_read = 0
        last_read_line = None
        while lines_read < first_row and last_read_line != '':
            last_read_line = fp.readline()
            lines_read += 1

        lines = []
        while lines_read < last_row:
            this_line = fp.readline()
            if this_line == '':
                break
            lines.append(this_line)
            lines_read += 1
    output = "[" + ", ".join(lines) + "]"
    res = make_response(output)
    res.mimetype = "application/json"
    return res

if __name__ == '__main__':
    if not sys.argv:
        print('Please provide the JSON file as the last argument')
        sys.exit(1)
    app.config['json_file'] = sys.argv[-1]
    app.run()
