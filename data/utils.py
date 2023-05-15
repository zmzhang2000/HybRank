import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, vstack

import transformers
transformers.logging.set_verbosity_error()

from pyserini.index.lucene import IndexReader
from pyserini.vectorizer import BM25Vectorizer
from pyserini.search import FaissSearcher, QueryEncoder


class BM25SimComputerCached(BM25Vectorizer):
    """Faster than BM25SimComputer when involve the same doc multiple times"""
    def __init__(self, lucene_index_path: str, min_df: int = 1, verbose: bool = False):
        super().__init__(lucene_index_path, min_df, verbose)
        self.index_reader = IndexReader(lucene_index_path)
        self.tf_vector_cache = dict()
        self.bm25_vector_cache = dict()

    def compute(self, q: str, docids: List[str], norm: Optional[str] = None):
        query_doc_matrix = np.array([[self.index_reader.compute_query_document_score(docid, q) for docid in docids]])
        doc_tf_vectors = self.get_tf_vectors(docids, norm)
        doc_bm25_vectors = self.get_bm25_vectors(docids, norm)
        doc_doc_matrix = doc_tf_vectors.dot(doc_bm25_vectors.T).toarray()
        return query_doc_matrix, doc_doc_matrix

    def get_tf_vectors(self, docids: List[str], norm: Optional[str] = None):
        """Get the tf vectors given a list of docids

        Parameters
        ----------
        norm : str
            Normalize the sparse matrix
        docids : List[str]
            The piece of text to analyze.

        Returns
        -------
        csr_matrix
            Sparse matrix representation of tf vectors
        """
        num_docs = len(docids)

        vectors = []
        for doc_id in docids:
            if doc_id in self.tf_vector_cache:
                vector = self.tf_vector_cache[doc_id]
            else:
                matrix_col, matrix_data = [], []
                # Term Frequency
                tf = self.index_reader.get_document_vector(doc_id)
                if tf is None:
                    vector = csr_matrix((1, self.vocabulary_size))
                else:
                    # Filter out in-eligible terms
                    tf = {t: tf[t] for t in tf if t in self.term_to_index}
                    # Convert from dict to sparse matrix
                    for term in tf:
                        tfidf = tf[term]
                        matrix_col.append(self.term_to_index[term])
                        matrix_data.append(tfidf)
                    matrix_row = [0] * len(matrix_col)
                    vector = csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(1, self.vocabulary_size))
                self.tf_vector_cache[doc_id] = vector

            vectors.append(vector)

        assert num_docs == len(vectors)
        vectors = vstack(vectors)
        if norm:
            return normalize(vectors, norm=norm)
        return vectors

    def get_bm25_vectors(self, docids: List[str], norm: Optional[str] = None):
        """Get the BM25 vectors given a list of docids

        Parameters
        ----------
        norm : str
            Normalize the sparse matrix
        docids : List[str]
            The piece of text to analyze.

        Returns
        -------
        csr_matrix
            Sparse matrix representation of BM25 vectors
        """
        num_docs = len(docids)

        vectors = []
        for doc_id in docids:
            if doc_id in self.bm25_vector_cache:
                vector = self.bm25_vector_cache[doc_id]
            else:
                matrix_col, matrix_data = [], []
                # Term Frequency
                tf = self.index_reader.get_document_vector(doc_id)
                if tf is None:
                    vector = csr_matrix((1, self.vocabulary_size))
                else:
                    # Filter out in-eligible terms
                    tf = {t: tf[t] for t in tf if t in self.term_to_index}
                    # Convert from dict to sparse matrix
                    for term in tf:
                        bm25_weight = self.index_reader.compute_bm25_term_weight(doc_id, term, analyzer=None)
                        matrix_col.append(self.term_to_index[term])
                        matrix_data.append(bm25_weight)
                    matrix_row = [0] * len(matrix_col)
                    vector = csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(1, self.vocabulary_size))
                self.bm25_vector_cache[doc_id] = vector

            vectors.append(vector)

        assert num_docs == len(vectors)
        vectors = vstack(vectors)
        if norm:
            return normalize(vectors, norm=norm)
        return vectors


class BM25SimComputer(BM25Vectorizer):
    def __init__(self, lucene_index_path: str, min_df: int = 1, verbose: bool = False):
        super().__init__(lucene_index_path, min_df, verbose)
        self.index_reader = IndexReader(lucene_index_path)

    def compute(self, q: str, docids: List[str], norm: Optional[str] = None):
        query_doc_matrix = np.array([[self.index_reader.compute_query_document_score(docid, q) for docid in docids]])
        doc_tf_vectors = self.get_tf_vectors(docids, norm)
        doc_bm25_vectors = self.get_bm25_vectors(docids, norm)
        doc_doc_matrix = doc_tf_vectors.dot(doc_bm25_vectors.T).toarray()
        return query_doc_matrix, doc_doc_matrix

    def get_tf_vectors(self, docids: List[str], norm: Optional[str] = None):
        """Get the tf vectors given a list of docids

        Parameters
        ----------
        norm : str
            Normalize the sparse matrix
        docids : List[str]
            The piece of text to analyze.

        Returns
        -------
        csr_matrix
            Sparse matrix representation of tf vectors
        """
        matrix_row, matrix_col, matrix_data = [], [], []
        num_docs = len(docids)

        for index, doc_id in enumerate(docids):
            # Term Frequency
            tf = self.index_reader.get_document_vector(doc_id)
            if tf is None:
                continue

            # Filter out in-eligible terms
            tf = {t: tf[t] for t in tf if t in self.term_to_index}

            # Convert from dict to sparse matrix
            for term in tf:
                tfidf = tf[term]
                matrix_row.append(index)
                matrix_col.append(self.term_to_index[term])
                matrix_data.append(tfidf)

        vectors = csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(num_docs, self.vocabulary_size))

        if norm:
            return normalize(vectors, norm=norm)
        return vectors

    def get_bm25_vectors(self, docids: List[str], norm: Optional[str] = None):
        """Get the BM25 vectors given a list of docids

        Parameters
        ----------
        norm : str
            Normalize the sparse matrix
        docids : List[str]
            The piece of text to analyze.

        Returns
        -------
        csr_matrix
            Sparse matrix representation of BM25 vectors
        """
        matrix_row, matrix_col, matrix_data = [], [], []
        num_docs = len(docids)

        for index, doc_id in enumerate(docids):

            # Term Frequency
            tf = self.index_reader.get_document_vector(doc_id)
            if tf is None:
                continue

            # Filter out in-eligible terms
            tf = {t: tf[t] for t in tf if t in self.term_to_index}

            # Convert from dict to sparse matrix
            for term in tf:
                bm25_weight = self.index_reader.compute_bm25_term_weight(doc_id, term, analyzer=None)
                matrix_row.append(index)
                matrix_col.append(self.term_to_index[term])
                matrix_data.append(bm25_weight)

        vectors = csr_matrix((matrix_data, (matrix_row, matrix_col)), shape=(num_docs, self.vocabulary_size))

        if norm:
            return normalize(vectors, norm=norm)
        return vectors


class DenseSimComputer(FaissSearcher):
    def __init__(self, index_dir: str, query_encoder: Union[QueryEncoder, str],
                 prebuilt_index_name: Optional[str] = None):
        super().__init__(index_dir, query_encoder, prebuilt_index_name)
        self.docid2idx = {docid: i for i, docid in enumerate(self.docids)}

    def compute(self, q: str, docids: List[str], norm: Optional[str] = None):
        query_vectors = self.get_query_vectors(q, norm)
        doc_vectors = self.get_doc_vectors(docids, norm)
        query_doc_matrix = query_vectors @ doc_vectors.T
        doc_doc_matrix = doc_vectors @ doc_vectors.T
        return query_doc_matrix, doc_doc_matrix

    def get_query_vectors(self, q: str, norm: Optional[str] = None):
        """Get the dense vectors given a query text

        Parameters
        ----------
        norm : str
            Normalize the matrix
        q : str
            The raw query text to analyze.

        Returns
        -------
        vectors
            Numpy matrix representation of dense vectors
        """
        vectors = self.query_encoder.encode(q)
        vectors = vectors.reshape((1, -1))

        if norm:
            return normalize(vectors, norm=norm)
        return vectors

    def get_doc_vectors(self, docids: List[str], norm: Optional[str] = None):
        """Get the dense vectors given a list of docids

        Parameters
        ----------
        norm : str
            Normalize the matrix
        docids : List[str]
            The piece of text to analyze.

        Returns
        -------
        vectors
            Numpy matrix representation of dense vectors
        """
        indices = [self.docid2idx[docid] for docid in docids]
        vectors = [self.index.reconstruct(i) for i in indices]
        vectors = np.vstack(vectors)

        if norm:
            return normalize(vectors, norm=norm)
        return vectors
