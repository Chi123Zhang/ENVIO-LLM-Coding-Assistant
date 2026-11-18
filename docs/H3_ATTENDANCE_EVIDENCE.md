# Attendance evidence and reviewed H3 comparisons

This private workflow extends original-source reconstruction. It does not change
the existing interview-coding pages or expose research uploads on the public app.

1. Rebuild H3 inputs from the original workbooks, CDC tables and attendance CSVs.
2. In **Find evidence for missing attendance**, upload original meeting PDFs.
   Filenames must identify one county and a meeting date. The date is a retrieval
   filter and still needs substantive verification against the document.
3. Choose **Retrieve attendance evidence for current gaps**. The existing ENVIO
   BM25 tokenizer and ranking retrieve text within the exact county and exposure
   month. Source memory records page, extracted-text line, content hash and PDF
   hash. Lines refer to extracted text, not printed PDF line numbers.
4. Download the review package. `A_independent_reviewer.csv` leaves answers and
   reviewer fields blank. `review_queue.csv` and `retrieval_log.json` contain the
   system-side records separately. Neither retrieval scores nor speaker counts
   establish a headcount. Only original PDF text enters source memory.
5. A researcher may return a completed review CSV. Approved records require the
   current run ID, an identified human reviewer, confirmed measurement definition,
   a reason and primary source support for each observed meeting. Set
   `meeting_counts_json` to a list of objects with `date`, `count`, `chunk_id`
   and `quote`. Count is a nonnegative integer and quote must be an exact source
   substring for the same county and meeting date. This validates provenance,
   not the truth of the human assessment. Do not relabel AI output as human review.
6. **Compare reviewed additions with the original data** runs the same existing
   exploratory H3 specification on both inputs. Only missing attendance changes.
   Monthly attendance is the mean of the reviewed meeting headcounts. Original
   files remain unchanged. The export preserves both analyses and a change ledger.

No approved records means no new before/after statistical comparison. Changing
source inputs, evidence PDFs or review content invalidates the relevant stored
result. Local model generation is a separate operator-run experiment; this web
retrieval feature makes no external or local LLM calls.

Independent evaluation is still required. Original and derived estimates alone
cannot measure value-recovery accuracy, review time or the causal effect of RAG.
