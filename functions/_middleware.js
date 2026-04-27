const BACKEND = "https://research-ai-agent-ih8r.onrender.com";

const API_PATHS = ["/research", "/config", "/jobs", "/static"];

export async function onRequest(context) {
  const { request, next } = context;
  const url = new URL(request.url);
  const path = url.pathname;

  const isApi = API_PATHS.some(p => path.startsWith(p));

  if (isApi) {
    const backendUrl = BACKEND + path + url.search;
    const proxyReq = new Request(backendUrl, {
      method: request.method,
      headers: request.headers,
      body: ["GET", "HEAD"].includes(request.method) ? undefined : request.body,
      redirect: "follow",
    });
    const resp = await fetch(proxyReq);
    const newHeaders = new Headers(resp.headers);
    newHeaders.set("Access-Control-Allow-Origin", "*");
    return new Response(resp.body, {
      status: resp.status,
      statusText: resp.statusText,
      headers: newHeaders,
    });
  }

  return next();
}
