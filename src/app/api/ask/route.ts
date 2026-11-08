
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    if (
      typeof body.question !== "string" ||
      !body.question.trim()
    ) {
      return NextResponse.json(
        { detail: "Question is required." },
        { status: 400 },
      );
    }

    const backendUrl =
      process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

    const response = await fetch(`${backendUrl}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: body.question.trim(),
      }),
      cache: "no-store",
    });

    const data = await response.json();

    return NextResponse.json(data, {
      status: response.status,
    });
  } catch {
    return NextResponse.json(
      { detail: "Unable to reach the backend API." },
      { status: 502 },
    );
  }
}
