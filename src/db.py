from io import TextIOWrapper
import tarfile
import json
import traceback
import threading
import queue
import mysql.connector
from clint.textui import progress


def iterate_tar(archive):
    """
    Iterate through members of the tar archive.

    Args:
        archive: Instance of tarfile.TarFile which should be iterated through.

    Yields:
        Instance of TarInfo, one for each member.
    """
    member = archive.next()
    while member is not None:
        yield member
        member = archive.next()


def iterate_contentdata(tar_path, verbose=True):
    """
    Iterate through the content files in the contentdata.tar.gz.

    Args:
        tar_path: Path to the contentdata.tar.gz file.
        verbose: Whether to print progress information.

    Yields:
        Dictionary with fields from this contentdata entry (article).
    """
    if not tarfile.is_tarfile(tar_path):
        raise ValueError("{tar_path} is not readable by tarfile module"
                         .format(tar_path=tar_path))

    if verbose:
        print('Opening tar file {tar_path'.format(tar_path=tar_path))

    with tarfile.open(tar_path) as archive:
        if verbose:
            print('Calculating number of members…')
        total_num = len(archive.getmembers())
        num_errors = 0
        num_processed = 0

        for member in progress.bar(
                iterate_tar(archive),
                expected_size=total_num,
                hide=not verbose
        ):
            num_processed += 1
            if not member.isfile():
                # Skip non-files
                continue
            if member.name.endswith('.data'):
                # Skip JSON archives (included in archive by error?)
                continue
            # TODO: Reduce number of indents (abstract away some of this)
            try:
                with archive.extractfile(member) as file:
                    # Go from io.BufferedReader to io.TextIO
                    with TextIOWrapper(file, encoding='utf8') as textfile:
                        # Process the first line of JSON
                        first_line = textfile.readline().strip()
                        parsed = json.loads(first_line)
                        # Make it possible to easily look up a specific field,
                        # by making the list of pairs into a dictionary
                        fields = parsed.get('fields', [])
                        as_dict = {field.get('field'): field.get('value')
                                   for field in fields}
                        as_dict['id'] = parsed.get('id')
                        as_dict['score'] = parsed.get('score')

                        yield as_dict
            except Exception:
                # One or two errors shouldn't mean you must re-run everything.
                # Still, print the error.
                traceback.print_exc()
                num_errors += 1
                # ...but abort if there are too many
                if num_errors >= 15 and (num_errors / num_processed) > 0.3:
                    print('Too many errors, giving up')
                    raise


class FulltextDb:
    def __init__(self, query_bundling_size=50, query_queue_size=150, **kwargs):
        """
        Create new instance of FulltextDb, capable of adding and quering
        fulltext metadata.

        Args:
            query_bundling_size: Number of queries to execute at once when
                inserting/updating the database.
            query_queue_size: Maximum number of queries the update_func can be
                ahead of the DB worker.
            **kwargs: Connection details given to mysql.connector.connect.
        """
        self._conn = mysql.connector.connect(**kwargs)
        self._insert_queue = queue.Queue(maxsize=query_queue_size)
        self._connection_details = kwargs
        self._query_bundling_size = query_bundling_size
        self._worker = None

    def get(self, document_id, columns=None):
        cursor = self._conn.cursor(dictionary=True)
        if columns is None:
            column_str = '*'
        else:
            column_str = ', '.join(columns)
        query = 'SELECT {columns} FROM article WHERE id=%s'.format(
            columns=column_str
        )
        cursor.execute(query, (document_id,))
        cursor.commit()
        for row in cursor:
            return row
        else:
            raise ValueError('No entry found for document id ' + document_id)

    def update(self, tar_path, columns, update_func, verbose=True):
        """
        Analyze the fulltext information and update the entries in the database.

        Args:
            tar_path: Path to the contentdata.tar.gz file.
            columns: The columns which update_func will return for each
                iteration. It is assumed the same columns will be returned for
                all calls to update_func within the same call to update().
            update_func: Function which takes in a dictionary of information
                from a file in contentdata.tar.gz (where key is the field name,
                and value is the field value) and returns a dictionary where
                key corresponds to database column and value is the value to
                set that column to. Return None to skip.
            verbose: Print progress information to stdout when set to True.
        """
        if 'id' not in columns:
            columns.append('id')

        query = self._create_insert_query(columns)
        self._start_db_worker(query)

        for fulltext in iterate_contentdata(tar_path, verbose):
            new_data = update_func(fulltext)
            if new_data is None:
                continue
            new_data['id'] = fulltext['id']
            self._add_insert_query(new_data, columns)

        self._stop_db_worker()

    def _add_insert_query(self, data, columns):
        """
        Add this query to the queue so it can be handled by the db_worker.

        Args:
            data: Dictionary of data to insert. Key is column name, value is the
                value.
            columns: Name of columns to insert. Must equal the columns given
                to self.create_query.
        """
        our_data = [data[col] for col in columns]
        self._insert_queue.put(our_data)

    def _start_db_worker(self, query):
        """
        Start the DB worker, which inserts/updates the DB.

        Only one DB worker can be running at any time.

        Args:
            query: The query to run.
        """
        if self._worker is not None:
            raise RuntimeError('Only one worker is supported at a time.')

        self._worker = threading.Thread(target=self._db_worker, kwargs={
            'connection_details': self._connection_details,
            'query_queue': self._insert_queue,
            'query': query,
            'query_bundling_size': self._query_bundling_size,
        })
        self._worker.start()

    @staticmethod
    def _db_worker(connection_details, query_queue, query, query_bundling_size):
        """
        Do the work of inserting into the database.

        Args:
            connection_details: Details to give to mysql.connector.connect.
            query_queue: Queue with parameters to run with query.
            query: The query to run, with placeholders. Will be run for every
                entry set into the query_queue.
            query_bundling_size: Number of queries to save up before executing
                them as one big query in MySQL.
        """
        conn = mysql.connector.connect(**connection_details)
        cursor = conn.cursor()

        bundled_queries = []

        def run_bundle():
            try:
                num = len(bundled_queries)
                cursor.executemany(query, bundled_queries)
                conn.commit()
                bundled_queries.clear()
                for _ in range(num):
                    query_queue.task_done()
            except mysql.connector.Error:
                traceback.print_exc()

        while True:
            params = query_queue.get()
            if params is None:
                break
            bundled_queries.append(params)
            if len(bundled_queries) >= query_bundling_size:
                run_bundle()

        if bundled_queries:
            run_bundle()

    def _stop_db_worker(self):
        """
        Ask and wait for the DB worker to terminate.
        """
        self._insert_queue.put(None)
        if self._worker is not None:
            self._worker.join()
            self._worker = None

    @staticmethod
    def _create_insert_query(columns):
        """
        Create insertion-update query for inserting/updating the given columns.

        Args:
            columns: List of columns which should be inserted into the
                database. Their order will determine the order of parameters
                given to add_query.

        Returns:
            str The generated query, which can be given to the DB worker.
        """
        whitelisted_columns = [
            'id',
            'word_count',
            'published',
        ]
        query_format = (
            "INSERT INTO articles ({columns}) VALUES ({placeholders}) "
            "ON DUPLICATE KEY UPDATE {update_clause}"
        )
        all_columns = [col for col in columns if col in whitelisted_columns]
        if 'id' not in all_columns:
            raise ValueError('The column id must be provided')

        if len(all_columns) != len(columns):
            raise ValueError('Some columns were not recognized')

        columns = ', '.join(all_columns)

        placeholders_list = ['%s'] * len(all_columns)
        placeholders = ', '.join(placeholders_list)

        updates = ['{col}=VALUES({col})'.format(col=col) for col in all_columns]
        update_clause = ', '.join(updates)

        query = query_format.format(
            columns=columns,
            placeholders=placeholders,
            update_clause=update_clause
        )
        return query

    def close(self):
        """
        Close the underlying database connection.
        """
        self._conn.close()
