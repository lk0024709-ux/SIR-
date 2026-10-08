"""Simple HTTP server for SIR AI Guru chat endpoint.

Uses the existing teachers/pipeline infrastructure with recorded transcripts.
"""

import json
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configuration
GRAPH_PATH = Path("data/curriculum/curriculum_graph.yaml")
TRANSCRIPT_PATH = Path("data/teacher_transcripts/hindi_teacher.jsonl")
CONFIG_PATH = Path("configs/curriculum_sampling.yaml")

# Load pipeline on module import - this sets graph, registry, pipeline
from curriculum.graph import CurriculumGraph
from teachers.providers import ProviderRegistry, RecordedProvider, TemplateProvider
from teachers.pipeline import TeacherPipeline, GenerationConfig
from sir_paths import REPO_ROOT, load_config, resolve


def load_pipeline_infrastructure():
    """Load the curriculum graph, provider registry, and pipeline."""
    graph = CurriculumGraph.load(resolve(GRAPH_PATH), validate=True)

    # Load registry with recorded transcripts
    registry = ProviderRegistry()
    if TRANSCRIPT_PATH.exists():
        provider = RecordedProvider.from_jsonl(TRANSCRIPT_PATH)
        registry.add(provider)
        print(f"Loaded recorded provider from {TRANSCRIPT_PATH}")

    # Add template provider for fallback
    registry.add(TemplateProvider())

    # Load config
    cfg = GenerationConfig.from_config(load_config(resolve(CONFIG_PATH)))

    # Create pipeline
    pipeline = TeacherPipeline(graph, registry, cfg)

    return graph, registry, pipeline


graph, registry, pipeline = load_pipeline_infrastructure()


def process_ai_guru_message(message):
    """Process a message using the AI Guru agent logic with the pipeline.

    Uses the existing curriculum pipeline to generate educational responses.
    Falls back to template provider if no recorded transcript answers are available.
    """
    message_lower = message.lower().strip()

    # Check if message is Hindi-related
    is_hindi = any(
        word in message_lower for word in ["hindi", "हindi", "हिन्दी", "वर्णमाला", "स्वर", "व्यंजन"]
    )

    # Check if message is math-related
    is_math = any(
        word in message_lower for word in ["percentage", "percent", "fractions", "mathematics", "g01", "g02"]
    )

    # Check if message is about specific curriculum node
    node_reference = None
    for node_id in graph.order():
        node = graph.nodes[node_id]
        if node.subject == "hindi" and node.grade is not None:
            topics = " ".join(node.topics)
            if any(t in message_lower for t in topics[:3]):
                node_reference = node_id
                break

    # Generate response using pipeline
    try:
        # Determine node and kind based on message content
        if is_hindi or node_reference:
            # Use Hindi foundation nodes
            hindi_nodes = [nid for nid in graph.by_subject("hindi") if graph.nodes[nid].grade is not None]
            if hindi_nodes:
                # Start from foundational node
                start_node = hindi_nodes[0]
                kinds = ["learn", "practice"]
                language = "hi"
                difficulty = "beginner"

                # Generate request
                from teachers.pipeline import build_request
                from teachers.providers import GenerationRequest

                request = build_request(
                    graph, start_node, kinds[0], language, difficulty, seed=0
                )

                # Generate drafts
                drafts = registry.generate(request)

                if drafts:
                    # Apply critics
                    from teachers.critics import critique_draft
                    from teachers.consensus import reach_consensus, worst_verdict

                    survivors = []
                    for draft in drafts:
                        critiques = critique_draft(
                            draft,
                            language=request.language,
                            required_fields=("task", "response", "answer"),
                            prompt_text=request.prompt,
                        )
                        verdict = worst_verdict(critiques)
                        if verdict != "reject":
                            survivors.append(draft)

                    if survivors:
                        # Reach consensus
                        outcome = reach_consensus(
                            survivors,
                            request_id=request.request_id,
                            mode="manual",
                            expected=None,
                            min_teachers=2,
                            allow_single_teacher=False,
                        )

                        # Create record
                        record = pipeline._record_from(
                            request, survivors, outcome,
                            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        )

                        # Build response from the task
                        task = record.task
                        response = record.response
                        answer = record.answer

                        # Format educational response
                        if task and response:
                            return f"""{task}

उत्तर: {answer}

मैं SIR का AI गुरु हूं। क्या आपको इस अवधारणा के बारे में और स्पष्टीकरण चाहिए या कोई अन्य प्रश्न है?"""

        # Default educational response
        return """मैं SIR का AI गुरु हूं। मुझे बताएं कि आप क्या सीखना चाहते हैं।

मैं हिंदी व्याकरण, गणित (percentage, fractions), या अन्य विषयों में मदद कर सकता हूं।

कृपया मुझे बताएं:
- किस विषय के बारे में जानना चाहते हैं?
- किस कक्षा स्तर पर?
- कोई विशिष्ट प्रश्न है?

मेरा लक्ष्य आपको estructured तरीके से सीखने में मदद करना है।"""

    except Exception as e:
        # Log error but don't crash
        print(f"AI Guru processing error: {e}")
        return """मैं SIR का AI गुरु हूं। थोड़ी देर में कोशिश करें।

कृपया फिर से प्रश्न पूछें या मुझे बताएं कि मैं आपकी किस तरह मदद कर सकता हूं।"""


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the AI Guru chat."""

    def do_GET(self):
        # Extract path without query string
        path = self.path.split("?")[0] if self.path else "/"

        if path == "/":
            # Serve the chat HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("/home/user/SIR-/frontend/chat.html", "rb") as f:
                self.wfile.write(f.read())
        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            uptime = time.time() - getattr(ChatHandler, "server_start_time", time.time())
            # Count hindi nodes
            hindi_count = len(graph.by_subject("hindi"))
            status = {
                "status": "ready",
                "message": "AI Guru is ready",
                "version": "P2-alpha",
                "uptime_seconds": int(uptime),
                "hindi_nodes": hindi_count,
                "pipeline_status": "operational",
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        # Extract path without query string
        path = self.path.split("?")[0] if self.path else "/"

        if path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                user_message = data.get("message", "")

                # Process using AI Guru agent
                response = process_ai_guru_message(user_message)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"response": response}).encode())
            except Exception as e:
                print(f"Chat handler error: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass


# Module-level server start time
ChatHandler.server_start_time = time.time()


def run_server(host="0.0.0.0", port=8080):
    """Run the HTTP server."""
    server = HTTPServer((host, port), ChatHandler)
    print(f"SIR AI Guru running on http://{host}:{port}")
    print(f"  - Hindi curriculum: {len(graph.by_subject('hindi'))} nodes")
    print(f"  - Pipeline: {len(graph.nodes)} total nodes across all subjects")
    print(f"  - Endpoints: GET /, GET /api/status, POST /api/chat")
    server.serve_forever()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    else:
        print("Server module loaded.")
        print("Usage: python server.py server  (starts the AI Guru chat server)")
        print("  Visit http://localhost:8080 to use the Web Chat UI")
        print("  API endpoints: GET /api/status, POST /api/chat")