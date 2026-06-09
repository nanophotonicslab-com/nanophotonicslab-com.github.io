import { EmailMessage } from "cloudflare:email";

const ALLOWED_ORIGINS = [
  "https://nanophotonicslab.com",
  "https://www.nanophotonicslab.com",
];

export default {
  async fetch(request, env) {
    const corsHeaders = getCorsHeaders(request);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== "POST") {
      return json({ error: "Method not allowed" }, 405, corsHeaders);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400, corsHeaders);
    }

    // Honeypot — bots fill hidden fields
    if (body._hp) {
      return json({ ok: true }, 200, corsHeaders);
    }

    const message = body.message?.trim();
    if (!message || message.length > 5000) {
      return json({ error: "Message required (max 5000 chars)" }, 400, corsHeaders);
    }

    const email = sanitizeEmail(body.email);
    const page = (body.page || "unknown").replace(/[\r\n]/g, "").slice(0, 500);

    // Build MIME email
    const headers = [
      "MIME-Version: 1.0",
      "From: noreply@nanophotonicslab.com",
      "To: info@nanophotonicslab.com",
    ];
    if (email) headers.push(`Reply-To: ${email}`);
    headers.push(
      "Subject: Website Feedback",
      "Content-Type: text/plain; charset=utf-8",
    );

    const bodyLines = [
      message,
      "",
      "---",
      email ? `Reply to: ${email}` : "No email provided",
      `Page: ${page}`,
    ];

    const rawEmail = headers.join("\r\n") + "\r\n\r\n" + bodyLines.join("\r\n");

    try {
      const msg = new EmailMessage(
        "noreply@nanophotonicslab.com",
        "joseramasa+nplab@gmail.com",
        new Response(rawEmail).body,
      );
      await env.SEND_EMAIL.send(msg);
      return json({ ok: true }, 200, corsHeaders);
    } catch {
      return json({ error: "Failed to send" }, 500, corsHeaders);
    }
  },
};

function json(data, status, corsHeaders) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

function getCorsHeaders(request) {
  const origin = request.headers.get("Origin") || "";
  const allowed = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    "Access-Control-Allow-Origin": allowed,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}

function sanitizeEmail(raw) {
  if (!raw) return "";
  const clean = raw.trim().replace(/[\r\n]/g, "").slice(0, 254);
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(clean) ? clean : "";
}
