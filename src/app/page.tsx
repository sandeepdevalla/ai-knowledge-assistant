
"use client";

import { FormEvent, useState } from "react";

type Source = {
  document: string;
  page: number;
};

type AskResponse = {
  answer: string;
  sources: Source[];
  model?: string;
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }

      const data: AskResponse = await response.json();
      setResult(data);
    } catch {
      setError(
        "Could not connect to the API. Check that FastAPI is running on port 8000 and that browser requests are allowed."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 px-5 py-12 text-zinc-900">
      <div className="mx-auto max-w-3xl">
        <header className="mb-10">
          <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-indigo-600">
            AI Engineering Project
          </p>
          <h1 className="text-4xl font-bold tracking-tight">
            AI Knowledge Assistant
          </h1>
          <p className="mt-3 text-zinc-600">
            Ask questions about your indexed documents and inspect the
            supporting sources.
          </p>
        </header>

        <form
          onSubmit={handleSubmit}
          className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm"
        >
          <label
            htmlFor="question"
            className="mb-2 block text-sm font-semibold"
          >
            Your question
          </label>

          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What are the five essential characteristics of cloud computing?"
            rows={3}
            required
            className="w-full resize-y rounded-xl border border-zinc-300 p-3 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100"
          />

          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="mt-4 rounded-xl bg-indigo-600 px-5 py-3 font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Searching documents..." : "Ask question"}
          </button>
        </form>

        {error && (
          <div
            role="alert"
            className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-red-700"
          >
            {error}
          </div>
        )}

        {result && (
          <section className="mt-8 space-y-5" aria-live="polite">
            <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold">Answer</h2>
              <p className="mt-4 whitespace-pre-wrap leading-7 text-zinc-700">
                {result.answer}
              </p>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm">
              <h2 className="text-lg font-semibold">
                Sources ({result.sources?.length ?? 0})
              </h2>

              {result.sources?.length ? (
                <ul className="mt-4 space-y-3">
                  {result.sources.map((source, index) => (
                    <li
                      key={`${source.document}-${source.page}-${index}`}
                      className="rounded-lg bg-zinc-50 p-3 text-sm"
                    >
                      <p className="font-medium">{source.document}</p>
                      <p className="mt-1 text-zinc-600">
                        PDF page {source.page}
                      </p>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-3 text-sm text-zinc-600">
                  No source documents were returned.
                </p>
              )}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
