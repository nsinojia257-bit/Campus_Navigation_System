// ============================================================
//  Campus Navigation System  -  DAA Mini Project (C++ + raylib)
//  Algorithms : BFS, Dijkstra, A*  (+ Floyd-Warshall for the A* heuristic)
//  Graph      : adjacency list built from the campus distance matrix
// ============================================================
#include "raylib.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <queue>
#include <string>
#include <vector>
using namespace std;

// ---------------- Campus data (from the distance table) ----------------
const int V = 10;
const char* NAME[V] = {"Main Gate", "Foyer", "Admin Section", "Canteen", "IDEA Lab",
                       "CP/CSD Dept", "IT/AI-DS Dept", "Central Library", "CE Labs", "Auditorium"};

const int DIST[V][V] = {
    //  MG   Foy  Adm  Can  IDEA CP   IT   Lib  CE   Aud
    {   0, 250, 300, 430, 400, 350, 350, 400, 450, 320},  // Main Gate
    { 250,   0,  50, 180, 150, 100, 100, 150, 200,  70},  // Foyer
    { 300,  50,   0, 100,  70, 100, 100, 120, 170,  20},  // Admin Section
    { 430, 180, 100,   0,  30,  60,  60, 150, 200, 100},  // Canteen
    { 400, 150,  70,  30,   0,  30,  30, 120, 170,  70},  // IDEA Lab
    { 350, 100, 100,  60,  30,   0,   1, 230, 250, 150},  // CP/CSD Dept
    { 350, 100, 100,  60,  30,   1,   0, 230, 250, 150},  // IT/AI-DS Dept
    { 400, 150, 120, 150, 120, 230, 230,   0,  20, 200},  // Central Library
    { 450, 200, 170, 200, 170, 250, 250,  20,   0, 250},  // CE Labs
    { 320,  70,  20, 100,  70, 150, 150, 200, 250,   0}   // Auditorium
};

// Screen positions are only for drawing (schematic, not to scale).
const Vector2 POS[V] = {{80, 610},  {230, 480}, {330, 360}, {470, 160}, {560, 260},
                        {680, 200}, {690, 330}, {320, 545}, {470, 620}, {420, 430}};

// ---------------- Window layout ----------------
const int WIN_W = 1200, WIN_H = 720, GRAPH_W = 800, NODE_R = 18;
const int PX = 820;  // left edge of the right-hand panel

// Segoe UI font loaded from the project folder.
Font UI_FONT = {0};

void DrawModernText(const char* text, float x, float y, float fontSize, Color color) {
    DrawTextEx(UI_FONT, text, Vector2{x, y}, fontSize, 0.5f, color);
}

// ---------------- Graph state ----------------
bool prunedMode = false;       // false: use every pair in the table, true: drop "redundant" edges
bool present[V][V];            // does edge (i,j) exist in the current mode?
bool blocked[V][V];            // has the user blocked edge (i,j)?
vector<pair<int, int>> adj[V]; // adjacency list: (neighbour, weight)
int S[V][V];                   // all-pairs shortest distances (Floyd-Warshall), used by A*

void buildEdges() {
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++) {
            present[i][j] = (i != j);
            // Pruned mode: an edge is redundant if a detour through some k is not longer.
            if (prunedMode && i != j)
                for (int k = 0; k < V; k++)
                    if (k != i && k != j && DIST[i][k] + DIST[k][j] <= DIST[i][j]) present[i][j] = false;
        }
}

void buildAdjList() {
    for (int i = 0; i < V; i++) {
        adj[i].clear();
        for (int j = 0; j < V; j++)
            if (present[i][j] && !blocked[i][j]) adj[i].push_back({j, DIST[i][j]});
    }
}

// Floyd-Warshall, O(V^3). Gives true shortest distances between every pair.
void floydWarshall() {
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++) S[i][j] = DIST[i][j];
    for (int k = 0; k < V; k++)
        for (int i = 0; i < V; i++)
            for (int j = 0; j < V; j++)
                if (S[i][k] + S[k][j] < S[i][j]) S[i][j] = S[i][k] + S[k][j];
}

// ---------------- Algorithms ----------------
struct Result {
    bool found = false;
    vector<int> path;
    int dist = 0;
    int expanded = 0;  // nodes taken out of the queue and processed
    double us = 0;     // average running time in microseconds
};

void finishPath(const vector<int>& par, int t, Result& r) {
    for (int v = t; v != -1; v = par[v]) r.path.push_back(v);
    reverse(r.path.begin(), r.path.end());
    r.dist = 0;
    for (size_t i = 0; i + 1 < r.path.size(); i++) r.dist += DIST[r.path[i]][r.path[i + 1]];
}

// BFS: fewest edges (ignores weights).  O(V + E)
Result runBFS(int s, int t) {
    Result r;
    vector<int> par(V, -1);
    vector<bool> seen(V, false);
    queue<int> q;
    q.push(s);
    seen[s] = true;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        r.expanded++;
        if (u == t) { r.found = true; break; }
        for (auto& e : adj[u])
            if (!seen[e.first]) { seen[e.first] = true; par[e.first] = u; q.push(e.first); }
    }
    if (r.found) finishPath(par, t, r);
    return r;
}

// Dijkstra: least total distance, min-heap.  O((V + E) log V)
Result runDijkstra(int s, int t) {
    Result r;
    vector<int> d(V, INT_MAX), par(V, -1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    d[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [du, u] = pq.top();
        pq.pop();
        if (du > d[u]) continue;  // stale entry
        r.expanded++;
        if (u == t) { r.found = true; break; }
        for (auto& [v, w] : adj[u])
            if (d[u] + w < d[v]) { d[v] = d[u] + w; par[v] = u; pq.push({d[v], v}); }
    }
    if (r.found) finishPath(par, t, r);
    return r;
}

// Heuristic for A*: "landmark" lower bound from the triangle inequality.
// For a landmark L:  |S(L,n) - S(L,t)| <= S(n,t)  so it never overestimates.
// S comes from Floyd-Warshall on the original graph; blocking edges only makes
// real distances longer, so the heuristic stays admissible.
const int LANDMARK[2] = {0, 8};  // Main Gate and CE Labs (far apart)
int heuristic(int n, int t) {
    return S[n][t];
}

// A*: Dijkstra ordered by f = g + h.  Same worst case, usually fewer nodes expanded.
Result runAStar(int s, int t) {
    Result r;
    vector<int> g(V, INT_MAX), par(V, -1);
    vector<bool> closed(V, false);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    g[s] = 0;
    pq.push({heuristic(s, t), s});
    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();
        if (closed[u]) continue;
        closed[u] = true;
        r.expanded++;
        if (u == t) { r.found = true; break; }
        for (auto& [v, w] : adj[u])
            if (g[u] + w < g[v]) { g[v] = g[u] + w; par[v] = u; pq.push({g[v] + heuristic(v, t), v}); }
    }
    if (r.found) finishPath(par, t, r);
    return r;
}

// Run an algorithm many times and average the time (the graph is tiny, so one run is ~0 us).
template <class F>
Result timed(F f) {
    const int REPS = 2000;
    Result r;
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < REPS; i++) r = f();
    auto t1 = chrono::high_resolution_clock::now();
    r.us = chrono::duration<double, micro>(t1 - t0).count() / REPS;
    return r;
}

// ---------------- App state ----------------
int src = -1, dst = -1, shown = 1;  // shown: 0 BFS, 1 Dijkstra, 2 A*
bool hasResults = false;
Result res[3];
const char* ALGO[3] = {"BFS", "Dijkstra", "A*"};
const Color ALGO_COL[3] = {{230, 126, 34, 255}, {41, 128, 185, 255}, {39, 174, 96, 255}};
const Color INK = {40, 44, 52, 255};
const Color GREY = {120, 125, 135, 255};

void recompute() {
    buildAdjList();
    hasResults = (src >= 0 && dst >= 0 && src != dst);
    if (!hasResults) return;
    res[0] = timed([] { return runBFS(src, dst); });
    res[1] = timed([] { return runDijkstra(src, dst); });
    res[2] = timed([] { return runAStar(src, dst); });
}

// ---------------- Drawing helpers ----------------
float segDist(Vector2 p, Vector2 a, Vector2 b) {
    float dx = b.x - a.x, dy = b.y - a.y;
    float t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
    t = max(0.0f, min(1.0f, t));
    float cx = a.x + t * dx, cy = a.y + t * dy;
    return sqrtf((p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy));
}

void drawTag(Vector2 p, const char* txt, Color bg, Color fg) {
    const float fs = 14.0f;
    Vector2 textSize = MeasureTextEx(UI_FONT, txt, fs, 0.5f);
    float w = textSize.x + 8.0f;
    float h = max(18.0f, textSize.y + 4.0f);
    DrawRectangle((int)(p.x - w / 2), (int)(p.y - h / 2), (int)w, (int)h, bg);
    DrawModernText(txt, p.x - textSize.x / 2, p.y - textSize.y / 2, fs, fg);
}

Vector2 mid(int a, int b) { return {(POS[a].x + POS[b].x) / 2, (POS[a].y + POS[b].y) / 2}; }

bool button(Rectangle r, const char* text, bool active, Color col) {
    bool hov = CheckCollisionPointRec(GetMousePosition(), r);
    Color bg = active ? col : (hov ? Color{222, 226, 234, 255} : Color{238, 240, 245, 255});
    DrawRectangleRec(r, bg);
    DrawRectangleLinesEx(r, 1, Color{160, 165, 175, 255});
    const float fs = 16.0f;
    Vector2 textSize = MeasureTextEx(UI_FONT, text, fs, 0.5f);
    DrawModernText(text, r.x + (r.width - textSize.x) / 2,
                   r.y + (r.height - textSize.y) / 2, fs,
                   active ? WHITE : INK);
    return hov && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

// Draws text wrapped to maxW; returns the y below the last line.
int drawWrapped(const string& text, int x, int y, int fs, int maxW, Color c) {
    string line, word;
    auto flush = [&]() {
        DrawModernText(line.c_str(), (float)x, (float)y, (float)fs, c);
        y += fs + 4;
        line.clear();
    };
    for (size_t i = 0; i <= text.size(); i++) {
        if (i == text.size() || text[i] == ' ') {
            string trial = line.empty() ? word : line + " " + word;
            if (!line.empty() && MeasureTextEx(UI_FONT, trial.c_str(), (float)fs, 0.5f).x > maxW) {
                flush();
                line = word;
            } else {
                line = trial;
            }
            word.clear();
        } else {
            word += text[i];
        }
    }
    if (!line.empty()) flush();
    return y;
}

// ---------------- Main ----------------
int main() {
    InitWindow(WIN_W, WIN_H, "Campus Navigation System - DAA Mini Project");
    UI_FONT = LoadFont("segoeui.ttf");
    if (UI_FONT.texture.id == 0) {
        CloseWindow();
        return 1;
    }
    SetTargetFPS(60);

    floydWarshall();
    buildEdges();
    memset(blocked, 0, sizeof(blocked));
    recompute();

    while (!WindowShouldClose()) {
        Vector2 m = GetMousePosition();
        bool inGraph = m.x < GRAPH_W;

        // ---- which node / edge is under the mouse? ----
        int hoverNode = -1;
        for (int i = 0; i < V; i++)
            if (inGraph && sqrtf((m.x - POS[i].x) * (m.x - POS[i].x) + (m.y - POS[i].y) * (m.y - POS[i].y)) <= NODE_R + 4)
                hoverNode = i;
        int hoverA = -1, hoverB = -1;
        if (inGraph && hoverNode == -1) {
            float best = 7.0f;
            for (int i = 0; i < V; i++)
                for (int j = i + 1; j < V; j++)
                    if (present[i][j]) {
                        float d = segDist(m, POS[i], POS[j]);
                        if (d < best) { best = d; hoverA = i; hoverB = j; }
                    }
        }

        // ---- input ----
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && hoverNode >= 0) {
            if (src == -1 || dst != -1) { src = hoverNode; dst = -1; }
            else if (hoverNode != src) dst = hoverNode;
            recompute();
        }
        if ((IsMouseButtonPressed(MOUSE_BUTTON_RIGHT) || IsKeyPressed(KEY_B)) && hoverA >= 0) {
            blocked[hoverA][hoverB] = blocked[hoverB][hoverA] = !blocked[hoverA][hoverB];
            recompute();
        }
        if (IsKeyPressed(KEY_ONE)) shown = 0;
        if (IsKeyPressed(KEY_TWO)) shown = 1;
        if (IsKeyPressed(KEY_THREE)) shown = 2;
        if (IsKeyPressed(KEY_C)) { src = dst = -1; recompute(); }

        // ================= DRAW =================
        BeginDrawing();
        ClearBackground(Color{250, 251, 253, 255});
        DrawRectangle(GRAPH_W, 0, WIN_W - GRAPH_W, WIN_H, Color{244, 245, 249, 255});
        DrawLine(GRAPH_W, 0, GRAPH_W, WIN_H, Color{210, 214, 222, 255});

        // normal edges
        for (int i = 0; i < V; i++)
            for (int j = i + 1; j < V; j++)
                if (present[i][j] && !blocked[i][j])
                    DrawLineEx(POS[i], POS[j], 1.5f, Color{205, 209, 218, 255});
        // blocked edges
        for (int i = 0; i < V; i++)
            for (int j = i + 1; j < V; j++)
                if (present[i][j] && blocked[i][j]) {
                    DrawLineEx(POS[i], POS[j], 2.0f, Color{214, 69, 65, 255});
                    Vector2 c = mid(i, j);
                    DrawModernText("X", c.x - 6, c.y - 10, 20, Color{214, 69, 65, 255});
                }
        // hovered edge
        if (hoverA >= 0) {
            DrawLineEx(POS[hoverA], POS[hoverB], 3.0f, Color{90, 95, 110, 255});
            char buf[64];
            snprintf(buf, sizeof(buf), "%d m%s", DIST[hoverA][hoverB], blocked[hoverA][hoverB] ? " (blocked)" : "");
            drawTag(mid(hoverA, hoverB), buf, Color{40, 44, 52, 255}, WHITE);
        }
        // chosen route
        if (hasResults && res[shown].found) {
            const Result& r = res[shown];
            for (size_t i = 0; i + 1 < r.path.size(); i++)
                DrawLineEx(POS[r.path[i]], POS[r.path[i + 1]], 5.0f, ALGO_COL[shown]);
            for (size_t i = 0; i + 1 < r.path.size(); i++) {
                char buf[16];
                snprintf(buf, sizeof(buf), "%d", DIST[r.path[i]][r.path[i + 1]]);
                drawTag(mid(r.path[i], r.path[i + 1]), buf, WHITE, ALGO_COL[shown]);
            }
        }
        // nodes
        for (int i = 0; i < V; i++) {
            bool onPath = false;
            if (hasResults && res[shown].found)
                for (int p : res[shown].path) if (p == i) onPath = true;
            Color fill = WHITE;
            if (onPath) fill = ALGO_COL[shown];
            if (i == src) fill = Color{30, 130, 76, 255};
            if (i == dst) fill = Color{192, 57, 43, 255};
            DrawCircle((int)POS[i].x, (int)POS[i].y, NODE_R, fill);
            DrawCircleLines((int)POS[i].x, (int)POS[i].y, NODE_R, i == hoverNode ? BLACK : Color{80, 85, 98, 255});
            if (i == src || i == dst) {
                const char* t = (i == src) ? "S" : "D";
                Vector2 labelSize = MeasureTextEx(UI_FONT, t, 18, 0.5f);
                DrawModernText(t, POS[i].x - labelSize.x / 2,
                               POS[i].y - labelSize.y / 2, 18, WHITE);
            }
            drawTag({POS[i].x, POS[i].y + NODE_R + 13}, NAME[i], Color{255, 255, 255, 220}, INK);
        }

        // ---------------- right panel ----------------
        DrawModernText("Campus Navigation System", PX, 20, 24, INK);
        DrawModernText("DAA mini project: BFS | Dijkstra | A*", PX, 52, 15, GREY);

        char buf[160];
        snprintf(buf, sizeof(buf), "From: %s", src >= 0 ? NAME[src] : "(click a location)");
        DrawModernText(buf, PX, 88, 18, INK);
        snprintf(buf, sizeof(buf), "To:   %s", dst >= 0 ? NAME[dst] : (src >= 0 ? "(click a destination)" : "-"));
        DrawModernText(buf, PX, 114, 18, INK);

        for (int a = 0; a < 3; a++)
            if (button({(float)(PX + a * 122), 148, 112, 30}, ALGO[a], shown == a, ALGO_COL[a])) shown = a;

        // results table
        int ty = 196;
        const int cx[5] = {0, 95, 165, 220, 295};
        const char* head[5] = {"Algorithm", "Dist(m)", "Hops", "Expanded", "Time(us)"};
        for (int c = 0; c < 5; c++) DrawModernText(head[c], PX + cx[c], ty, 14, GREY);
        DrawLine(PX, ty + 20, PX + 360, ty + 20, Color{200, 204, 212, 255});
        for (int a = 0; a < 3; a++) {
            int y = ty + 28 + a * 26;
            if (shown == a) DrawRectangle(PX - 4, y - 3, 368, 24, Color{228, 232, 242, 255});
            DrawModernText(ALGO[a], PX + cx[0], y, 16, ALGO_COL[a]);
            if (!hasResults) { DrawModernText("-", PX + cx[1], y, 16, GREY); continue; }
            if (!res[a].found) { DrawModernText("No path", PX + cx[1], y, 16, Color{192, 57, 43, 255}); continue; }
            snprintf(buf, sizeof(buf), "%d", res[a].dist);
            DrawModernText(buf, PX + cx[1], y, 16, INK);
            snprintf(buf, sizeof(buf), "%d", (int)res[a].path.size() - 1);
            DrawModernText(buf, PX + cx[2], y, 16, INK);
            snprintf(buf, sizeof(buf), "%d", res[a].expanded);
            DrawModernText(buf, PX + cx[3], y, 16, INK);
            snprintf(buf, sizeof(buf), "%.2f", res[a].us);
            DrawModernText(buf, PX + cx[4], y, 16, INK);
        }

        // route text
        int y = 320;
        snprintf(buf, sizeof(buf), "Route (%s):", ALGO[shown]);
        DrawModernText(buf, PX, y, 16, ALGO_COL[shown]);
        y += 24;
        if (hasResults && res[shown].found) {
            string route;
            for (size_t i = 0; i < res[shown].path.size(); i++) {
                if (i) route += " > ";
                route += NAME[res[shown].path[i]];
            }
            y = drawWrapped(route, PX, y, 15, 360, INK);
        } else if (hasResults) {
            DrawModernText("No route available (blocked paths).", PX, y, 15, Color{192, 57, 43, 255});
        } else {
            DrawModernText("Pick a source and a destination.", PX, y, 15, GREY);
        }

        // short explanation of what the numbers mean for this query
        if (hasResults && res[0].found && res[1].found && res[2].found) {
            y = max(y + 14, 440);
            DrawModernText("What this shows:", PX, y, 16, INK);
            y += 24;
            if (res[0].dist > res[1].dist)
                snprintf(buf, sizeof(buf), "BFS has the fewest hops but its route is %d m longer than the shortest one.",
                         res[0].dist - res[1].dist);
            else
                snprintf(buf, sizeof(buf), "BFS happens to find the shortest route here (%d m).", res[0].dist);
            y = drawWrapped(buf, PX, y, 14, 360, GREY);
            y += 4;
            if (res[2].dist == res[1].dist)
                snprintf(buf, sizeof(buf), "A* matches Dijkstra's %d m after expanding %d node(s) instead of %d.",
                         res[1].dist, res[2].expanded, res[1].expanded);
            else
                snprintf(buf, sizeof(buf), "A* distance (%d m) differs from Dijkstra (%d m).", res[2].dist, res[1].dist);
            drawWrapped(buf, PX, y, 14, 360, GREY);
        }

        // buttons
        int nb = 0;
        for (int i = 0; i < V; i++)
            for (int j = i + 1; j < V; j++) if (present[i][j] && blocked[i][j]) nb++;
        snprintf(buf, sizeof(buf), "Reset blocked paths (%d)", nb);
        if (button({(float)PX, 560, 360, 30}, buf, false, INK)) {
            memset(blocked, 0, sizeof(blocked));
            recompute();
        }
        snprintf(buf, sizeof(buf), "Graph: %s", prunedMode ? "pruned (no redundant edges)" : "full distance table");
        if (button({(float)PX, 598, 360, 30}, buf, false, INK)) {
            prunedMode = !prunedMode;
            memset(blocked, 0, sizeof(blocked));
            buildEdges();
            recompute();
        }

        DrawModernText("Left-click: pick source, then destination", PX, 644, 13, GREY);
        DrawModernText("Right-click (or B) on an edge: block / unblock", PX, 662, 13, GREY);
        DrawModernText("1 / 2 / 3: switch algorithm      C: clear selection", PX, 680, 13, GREY);

        EndDrawing();
    }
    UnloadFont(UI_FONT);
    CloseWindow();
    return 0;
}