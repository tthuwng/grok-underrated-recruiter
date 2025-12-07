import React, { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_BASE = import.meta.env.VITE_API_URL || "/api";

interface GraphNode {
  id: string;
  handle: string;
  name: string;
  bio: string;
  followers_count: number;
  following_count: number;
  is_seed: boolean;
  is_candidate: boolean;
  pagerank_score: number;
  underratedness_score: number;
  grok_relevant: boolean | null;
  grok_role: string | null;
  depth: number;
  discovered_via?: string;
  submission_pending?: boolean;
  x?: number;
  y?: number;
}

// Extended node details from /node/{id} endpoint
interface NodeDetails extends GraphNode {
  grok_reason: string | null;
  incoming_connections: number;
  outgoing_connections: number;
  x_url: string;
}

interface GraphEdge {
  source: string;
  target: string;
  weight: number;
  interaction_type: string;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_nodes: number;
    total_edges: number;
    displayed_nodes: number;
    displayed_edges: number;
    seeds: number;
    filtered_count: number;
  };
}

export function GraphView() {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>();
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [nodeDetails, setNodeDetails] = useState<NodeDetails | null>(null);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<GraphNode[]>([]);
  const [maxNodes, setMaxNodes] = useState(5000);
  const [maxDepth, setMaxDepth] = useState(5);
  const [onlyRelevant, setOnlyRelevant] = useState(false);

  const fetchGraph = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch(
        `${API_BASE}/graph?max_nodes=${maxNodes}&only_relevant=${onlyRelevant}&max_depth=${maxDepth}`
      );
      if (!response.ok) throw new Error("Failed to fetch graph");
      const data: GraphData = await response.json();
      setGraphData(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load graph");
    } finally {
      setIsLoading(false);
    }
  }, [maxNodes, onlyRelevant, maxDepth]);

  useEffect(() => {
    fetchGraph();
  }, [fetchGraph]);

  // Fetch detailed node info when a node is selected
  const fetchNodeDetails = useCallback(async (nodeId: string) => {
    try {
      setIsLoadingDetails(true);
      setNodeDetails(null);
      const response = await fetch(`${API_BASE}/node/${nodeId}`);
      if (!response.ok) throw new Error("Failed to fetch node details");
      const data: NodeDetails = await response.json();
      setNodeDetails(data);
    } catch (e) {
      console.error("Failed to fetch node details:", e);
    } finally {
      setIsLoadingDetails(false);
    }
  }, []);

  // Fetch node details when selectedNode changes
  useEffect(() => {
    if (selectedNode) {
      fetchNodeDetails(selectedNode.id);
    } else {
      setNodeDetails(null);
    }
  }, [selectedNode, fetchNodeDetails]);

  // Configure d3 forces for better spacing - scales with node count
  useEffect(() => {
    if (graphRef.current && graphData) {
      const fg = graphRef.current;
      const nodeCount = graphData.nodes.length;

      // Scale forces based on node count for better visualization
      // More nodes = weaker charge to prevent overcrowding, shorter links
      const chargeStrength =
        nodeCount > 1000 ? -150 : nodeCount > 500 ? -250 : -400;
      const linkDistance = nodeCount > 1000 ? 60 : nodeCount > 500 ? 80 : 120;
      const linkStrength = nodeCount > 1000 ? 0.1 : 0.2;

      fg.d3Force("charge")?.strength(chargeStrength).distanceMax(500);
      fg.d3Force("link")?.distance(linkDistance).strength(linkStrength);
      fg.d3Force("collide", null);
    }
  }, [graphData]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (!query || query.length < 2 || !graphData) {
      setSearchResults([]);
      return;
    }
    const q = query.toLowerCase().replace("@", "");
    const matches = graphData.nodes
      .filter(
        (n) =>
          n.handle.toLowerCase().includes(q) ||
          (n.name && n.name.toLowerCase().includes(q))
      )
      .slice(0, 8);
    setSearchResults(matches);
  };

  const focusOnNode = (nodeFromSearch: GraphNode) => {
    setSearchResults([]);
    setSearchQuery("");

    // Find the actual node in graphData (which has x/y coordinates from simulation)
    if (graphData && graphRef.current) {
      const actualNode = graphData.nodes.find(
        (n) => n.id === nodeFromSearch.id
      );
      if (actualNode) {
        console.log('Focus on node:', actualNode.handle, 'x:', actualNode.x, 'y:', actualNode.y);
        setSelectedNode(actualNode);

        // Check if coordinates are set
        if (actualNode.x === undefined || actualNode.y === undefined) {
          console.warn('Node coordinates not set - simulation may not have completed');
          return;
        }

        const fg = graphRef.current;

        // Get current zoom to calculate proper animation
        const currentZoom = fg.zoom();
        const targetZoom = 4;

        // If we're already zoomed in, zoom out first for better transition
        if (currentZoom > 2) {
          fg.zoom(1, 200);
          setTimeout(() => {
            fg.centerAt(actualNode.x, actualNode.y, 400);
            setTimeout(() => {
              fg.zoom(targetZoom, 400);
            }, 450);
          }, 250);
        } else {
          // Zoom out slightly first, then center, then zoom in
          fg.centerAt(actualNode.x, actualNode.y, 600);
          setTimeout(() => {
            fg.zoom(targetZoom, 400);
          }, 650);
        }
      } else {
        console.warn('Node not found in graph data:', nodeFromSearch.id);
      }
    }
  };

  const runGrokFilter = async () => {
    try {
      setIsLoading(true);
      await fetch(`${API_BASE}/filter`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "" }),
      });
      // Poll for completion
      const poll = async () => {
        try {
          const response = await fetch(`${API_BASE}/status`);
          const status = await response.json();
          if (status.loading) {
            setTimeout(poll, 1000);
          } else {
            if (status.last_error) {
              setError(status.last_error);
            }
            fetchGraph();
          }
        } catch {
          setError("Connection error");
          setIsLoading(false);
        }
      };
      poll();
    } catch (e) {
      setError("Failed to run filter");
      setIsLoading(false);
    }
  };

  const saveGraph = async () => {
    try {
      const response = await fetch(`${API_BASE}/save`, { method: "POST" });
      if (!response.ok) throw new Error("Failed to save");
      // Show brief success
      setError(null);
    } catch (e) {
      setError("Failed to save graph");
    }
  };

  const getNodeColor = (node: GraphNode) => {
    if (node.is_seed) return "#f87171"; // accent-red
    // Pending submission (yellow/amber)
    if (
      node.discovered_via === "user_submission" &&
      node.grok_relevant === null
    ) {
      return "#fbbf24"; // amber-400
    }
    if (node.grok_relevant === true) return "#34d399"; // accent-green
    if (node.grok_relevant === false) return "#3f3f46";
    if (node.is_candidate) return "#60a5fa"; // accent-blue
    return "#71717a"; // text-muted
  };

  const getNodeSize = (node: GraphNode) => {
    // Scale node sizes based on total node count for better dense graph visualization
    const nodeCount = graphData?.nodes.length || 200;
    const scaleFactor = nodeCount > 1000 ? 0.5 : nodeCount > 500 ? 0.7 : 1.0;

    if (node.is_seed) return 12 * scaleFactor;
    const pr = node.pagerank_score || 0;
    const baseSize = Math.max(3, Math.min(14, 3 + pr * 4000));
    return baseSize * scaleFactor;
  };

  // Get connected node IDs for highlighting edges - used in nodeCanvasObject callback below
  const connectedNodeIds = React.useMemo(() => {
    if (!selectedNode || !graphData) return new Set<string>();
    const ids = new Set<string>();
    graphData.edges.forEach(edge => {
      if (edge.source === selectedNode.id || edge.target === selectedNode.id) {
        ids.add(edge.source);
        ids.add(edge.target);
      }
    });
    return ids;
  }, [selectedNode, graphData]);

  // Check if an edge is connected to the selected node
  const isEdgeConnected = (source: string, target: string) => {
    if (!selectedNode) return false;
    return source === selectedNode.id || target === selectedNode.id;
  };

  // Custom node rendering with labels
  const nodeCanvasObject = React.useCallback((
    node: GraphNode & { x?: number; y?: number },
    ctx: CanvasRenderingContext2D,
    globalScale: number
  ) => {
    const size = getNodeSize(node);
    const color = getNodeColor(node);
    const x = node.x || 0;
    const y = node.y || 0;

    // Check if this node is connected to the selected node
    const isConnected = selectedNode && connectedNodeIds.has(node.id);
    const isSelected = selectedNode?.id === node.id;

    // Dim nodes that are not connected when a node is selected
    const dimNode = selectedNode && !isConnected && !isSelected;

    // Draw highlight ring for selected or connected nodes
    if (isSelected) {
      ctx.beginPath();
      ctx.arc(x, y, size + 4, 0, 2 * Math.PI, false);
      ctx.fillStyle = "rgba(96, 165, 250, 0.3)";
      ctx.fill();
      ctx.strokeStyle = "#60a5fa";
      ctx.lineWidth = 2;
      ctx.stroke();
    } else if (isConnected) {
      ctx.beginPath();
      ctx.arc(x, y, size + 2, 0, 2 * Math.PI, false);
      ctx.strokeStyle = "rgba(96, 165, 250, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Draw node circle
    ctx.beginPath();
    ctx.arc(x, y, size, 0, 2 * Math.PI, false);
    ctx.fillStyle = dimNode ? `${color}40` : color; // Add transparency if dimmed
    ctx.fill();

    // Draw white text label for seeds and high PageRank nodes, or connected/selected nodes
    const showLabel =
      node.is_seed ||
      node.pagerank_score > 0.001 ||
      node.grok_relevant === true ||
      isSelected ||
      isConnected;
    if (showLabel) {
      const label = `@${node.handle}`;
      const fontSize = Math.max(12 / globalScale, 4);
      ctx.font = `${fontSize}px -apple-system, BlinkMacSystemFont, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = dimNode ? "rgba(255, 255, 255, 0.3)" : "#ffffff";
      ctx.fillText(label, x, y + size + 2);
    }
  }, [selectedNode, connectedNodeIds, graphData]);

  return (
    <div style={styles.container}>
      <div style={styles.sidebar}>
        {/* Search */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Search Graph</div>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search by @handle..."
            style={styles.input}
          />
          {searchResults.length > 0 && (
            <div style={styles.searchResults}>
              {searchResults.map((node) => (
                <div
                  key={node.id}
                  style={styles.searchResult}
                  onClick={() => focusOnNode(node)}
                >
                  <div style={styles.searchResultHandle}>@{node.handle}</div>
                  <div style={styles.searchResultName}>{node.name}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Stats */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Statistics</div>
          <div style={styles.statsGrid}>
            <div style={styles.statCard}>
              <div style={styles.statValue}>
                {graphData?.nodes.length.toLocaleString() || 0}
              </div>
              <div style={styles.statLabel}>Displayed</div>
            </div>
            <div style={styles.statCard}>
              <div style={styles.statValue}>
                {graphData?.edges.length.toLocaleString() || 0}
              </div>
              <div style={styles.statLabel}>Edges</div>
            </div>
            <div style={styles.statCard}>
              <div style={styles.statValue}>{graphData?.stats.seeds || 0}</div>
              <div style={styles.statLabel}>Seeds</div>
            </div>
            <div style={styles.statCard}>
              <div style={styles.statValue}>
                {graphData?.stats.total_nodes.toLocaleString() || 0}
              </div>
              <div style={styles.statLabel}>Total</div>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Display</div>
          <div style={styles.field}>
            <label style={styles.label}>Max Nodes: {maxNodes.toLocaleString()}</label>
            <input
              type="range"
              min={500}
              max={20000}
              step={500}
              value={maxNodes}
              onChange={(e) => setMaxNodes(Number(e.target.value))}
              style={styles.range}
            />
          </div>
          <div style={{ ...styles.field, marginTop: '12px' }}>
            <label style={styles.label}>Max Depth: {maxDepth}</label>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={maxDepth}
              onChange={(e) => setMaxDepth(Number(e.target.value))}
              style={styles.range}
            />
          </div>
          <label style={styles.checkbox}>
            <input
              type="checkbox"
              checked={onlyRelevant}
              onChange={(e) => setOnlyRelevant(e.target.checked)}
            />
            Show only Grok-relevant
          </label>
        </div>

        {/* Actions */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Actions</div>
          <div style={styles.buttonGroup}>
            <button
              onClick={runGrokFilter}
              disabled={isLoading}
              style={{ ...styles.button, ...styles.buttonSecondary }}
            >
              Run Grok Filter
            </button>
            <button
              onClick={fetchGraph}
              disabled={isLoading}
              style={{ ...styles.button, ...styles.buttonSecondary }}
            >
              Refresh
            </button>
          </div>
          <div style={{ ...styles.buttonGroup, marginTop: "8px" }}>
            <button
              onClick={saveGraph}
              style={{ ...styles.button, ...styles.buttonSecondary }}
            >
              Save Graph
            </button>
          </div>
        </div>

        {/* Legend */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Legend</div>
          <div style={styles.legend}>
            <div style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: "#f87171" }} />
              Seeds
            </div>
            <div style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: "#fbbf24" }} />
              Pending evaluation
            </div>
            <div style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: "#34d399" }} />
              Grok relevant
            </div>
            <div style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: "#60a5fa" }} />
              Candidates
            </div>
            <div style={styles.legendItem}>
              <span style={{ ...styles.legendDot, background: "#71717a" }} />
              Other
            </div>
          </div>
          <div style={styles.legendNote}>Node size = PageRank score</div>
        </div>
      </div>

      {/* Graph Area */}
      <div style={styles.graphArea}>
        {error && <div style={styles.error}>{error}</div>}

        {isLoading && (
          <div style={styles.loadingOverlay}>
            <div style={styles.spinner} />
            <div>Loading graph...</div>
          </div>
        )}

        {graphData && (
          <ForceGraph2D
            ref={graphRef}
            graphData={{
              nodes: graphData.nodes,
              links: graphData.edges.map((e) => ({
                source: e.source,
                target: e.target,
              })),
            }}
            nodeId="id"
            nodeLabel={(node) =>
              `@${(node as GraphNode).handle}\n${
                (node as GraphNode).name || ""
              }`
            }
            nodeCanvasObject={(node, ctx, globalScale) =>
              nodeCanvasObject(
                node as GraphNode & { x?: number; y?: number },
                ctx,
                globalScale
              )
            }
            nodePointerAreaPaint={(node, color, ctx) => {
              const size = getNodeSize(node as GraphNode);
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(
                (node as GraphNode & { x: number }).x || 0,
                (node as GraphNode & { y: number }).y || 0,
                size + 2,
                0,
                2 * Math.PI
              );
              ctx.fill();
            }}
            linkColor={(link) => {
              // Get source and target IDs (can be objects or strings depending on simulation state)
              const sourceId = typeof link.source === 'object' ? (link.source as GraphNode).id : link.source;
              const targetId = typeof link.target === 'object' ? (link.target as GraphNode).id : link.target;

              // Highlight edges connected to selected node
              if (selectedNode && isEdgeConnected(sourceId as string, targetId as string)) {
                return "rgba(96, 165, 250, 0.8)"; // accent-blue with high opacity
              }

              const nodeCount = graphData?.nodes.length || 200;
              // Dim other edges when a node is selected
              const alpha = selectedNode
                ? 0.03
                : nodeCount > 1000 ? 0.08 : nodeCount > 500 ? 0.12 : 0.15;
              return `rgba(113, 113, 122, ${alpha})`;
            }}
            linkWidth={(link) => {
              // Get source and target IDs
              const sourceId = typeof link.source === 'object' ? (link.source as GraphNode).id : link.source;
              const targetId = typeof link.target === 'object' ? (link.target as GraphNode).id : link.target;

              // Make edges connected to selected node thicker
              if (selectedNode && isEdgeConnected(sourceId as string, targetId as string)) {
                return 2;
              }

              const nodeCount = graphData?.nodes.length || 200;
              return nodeCount > 1000 ? 0.3 : nodeCount > 500 ? 0.4 : 0.5;
            }}
            backgroundColor="#09090b"
            onNodeClick={(node) => setSelectedNode(node as GraphNode)}
            onBackgroundClick={() => setSelectedNode(null)}
            d3AlphaDecay={0.02}
            d3VelocityDecay={0.3}
            cooldownTicks={300}
            d3AlphaMin={0.001}
            warmupTicks={100}
            onEngineStop={() => graphRef.current?.zoomToFit(400, 50)}
          />
        )}

        {/* Floating Legend */}
        <div style={styles.floatingLegend}>
          <div style={styles.floatingLegendTitle}>Legend</div>
          <div style={styles.floatingLegendItems}>
            <div style={styles.floatingLegendItem}>
              <span
                style={{ ...styles.floatingLegendDot, background: "#f87171" }}
              />
              <span>Seeds</span>
            </div>
            <div style={styles.floatingLegendItem}>
              <span
                style={{ ...styles.floatingLegendDot, background: "#fbbf24" }}
              />
              <span>Pending</span>
            </div>
            <div style={styles.floatingLegendItem}>
              <span
                style={{ ...styles.floatingLegendDot, background: "#34d399" }}
              />
              <span>Grok Relevant</span>
            </div>
            <div style={styles.floatingLegendItem}>
              <span
                style={{ ...styles.floatingLegendDot, background: "#60a5fa" }}
              />
              <span>Candidates</span>
            </div>
            <div style={styles.floatingLegendItem}>
              <span
                style={{ ...styles.floatingLegendDot, background: "#71717a" }}
              />
              <span>Other</span>
            </div>
          </div>
          <div style={styles.floatingLegendNote}>Node size = PageRank</div>
        </div>

        {/* Node Details Panel */}
        {selectedNode && (
          <div style={styles.nodePanel}>
            <div style={styles.nodePanelHeader}>
              <div>
                <div style={styles.nodeHandle}>@{selectedNode.handle}</div>
                <div style={styles.nodeName}>{selectedNode.name}</div>
              </div>
              <button
                style={styles.closeButton}
                onClick={() => setSelectedNode(null)}
              >
                &times;
              </button>
            </div>
            <div style={styles.nodeBio}>{selectedNode.bio || "No bio"}</div>
            <div style={styles.nodeTags}>
              {selectedNode.is_seed && (
                <span style={{ ...styles.nodeTag, ...styles.tagSeed }}>
                  Seed
                </span>
              )}
              {selectedNode.discovered_via === "user_submission" &&
                selectedNode.grok_relevant === null && (
                  <span style={{ ...styles.nodeTag, ...styles.tagPending }}>
                    Pending
                  </span>
                )}
              {selectedNode.grok_relevant === true && (
                <span style={{ ...styles.nodeTag, ...styles.tagRelevant }}>
                  Relevant
                </span>
              )}
              {selectedNode.grok_role && (
                <span style={styles.nodeTag}>{selectedNode.grok_role}</span>
              )}
              {selectedNode.is_candidate && (
                <span style={styles.nodeTag}>Candidate</span>
              )}
            </div>

            {/* Grok Evaluation Reason */}
            {nodeDetails?.grok_reason && (
              <div style={styles.grokReasonSection}>
                <div style={styles.grokReasonLabel}>Grok Analysis</div>
                <div style={styles.grokReasonText}>
                  {nodeDetails.grok_reason}
                </div>
              </div>
            )}
            {isLoadingDetails && !nodeDetails && (
              <div style={styles.loadingDetails}>Loading details...</div>
            )}

            <div style={styles.nodeStats}>
              <div style={styles.nodeStat}>
                <div style={styles.nodeStatValue}>
                  {selectedNode.followers_count.toLocaleString()}
                </div>
                <div style={styles.nodeStatLabel}>Followers</div>
              </div>
              <div style={styles.nodeStat}>
                <div style={styles.nodeStatValue}>
                  {(selectedNode.pagerank_score * 1000).toFixed(3)}
                </div>
                <div style={styles.nodeStatLabel}>PageRank</div>
              </div>
              {nodeDetails && (
                <>
                  <div style={styles.nodeStat}>
                    <div style={styles.nodeStatValue}>
                      {nodeDetails.incoming_connections}
                    </div>
                    <div style={styles.nodeStatLabel}>In-Links</div>
                  </div>
                  <div style={styles.nodeStat}>
                    <div style={styles.nodeStatValue}>
                      {nodeDetails.outgoing_connections}
                    </div>
                    <div style={styles.nodeStatLabel}>Out-Links</div>
                  </div>
                </>
              )}
            </div>
            <a
              href={`https://x.com/${selectedNode.handle}`}
              target="_blank"
              rel="noopener noreferrer"
              style={styles.viewOnX}
            >
              View on X
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    height: "calc(100vh - 140px)",
    background: "var(--bg-primary)",
  },
  sidebar: {
    width: "300px",
    background: "var(--bg-secondary)",
    borderRight: "1px solid var(--border-color)",
    overflowY: "auto",
    flexShrink: 0,
  },
  section: {
    padding: "16px 20px",
    borderBottom: "1px solid var(--border-color)",
  },
  sectionTitle: {
    fontSize: "11px",
    fontWeight: 600,
    color: "var(--text-muted)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: "12px",
  },
  input: {
    width: "100%",
    padding: "10px 12px",
    background: "var(--bg-tertiary)",
    border: "1px solid var(--border-color)",
    borderRadius: "8px",
    color: "var(--text-primary)",
    fontSize: "14px",
    outline: "none",
  },
  row: {
    display: "flex",
    gap: "10px",
    marginTop: "10px",
  },
  field: {
    flex: 1,
  },
  label: {
    display: "block",
    fontSize: "12px",
    color: "var(--text-secondary)",
    marginBottom: "6px",
  },
  select: {
    width: "100%",
    padding: "8px 10px",
    background: "var(--bg-tertiary)",
    border: "1px solid var(--border-color)",
    borderRadius: "6px",
    color: "var(--text-primary)",
    fontSize: "13px",
  },
  button: {
    width: "100%",
    padding: "10px 16px",
    fontSize: "13px",
    fontWeight: 500,
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    marginTop: "12px",
    transition: "all 0.2s",
  },
  buttonPrimary: {
    background: "var(--accent-primary)",
    color: "#000",
  },
  buttonSecondary: {
    background: "var(--bg-tertiary)",
    color: "var(--text-secondary)",
    border: "1px solid var(--border-color)",
  },
  buttonDisabled: {
    background: "var(--bg-tertiary)",
    color: "var(--text-muted)",
    cursor: "not-allowed",
  },
  buttonGroup: {
    display: "flex",
    gap: "8px",
  },
  statsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: "10px",
  },
  statCard: {
    background: "var(--bg-tertiary)",
    borderRadius: "8px",
    padding: "12px",
  },
  statValue: {
    fontSize: "20px",
    fontWeight: 600,
    color: "var(--text-primary)",
  },
  statLabel: {
    fontSize: "10px",
    color: "var(--text-muted)",
    textTransform: "uppercase",
    marginTop: "2px",
  },
  range: {
    width: "100%",
    accentColor: "var(--accent-primary)",
  },
  checkbox: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "13px",
    color: "var(--text-secondary)",
    marginTop: "12px",
    cursor: "pointer",
  },
  legend: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "12px",
    color: "var(--text-secondary)",
  },
  legendDot: {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
  },
  legendNote: {
    fontSize: "11px",
    color: "var(--text-muted)",
    marginTop: "8px",
  },
  graphArea: {
    flex: 1,
    position: "relative",
    overflow: "hidden",
  },
  error: {
    position: "absolute",
    top: "16px",
    left: "16px",
    right: "16px",
    padding: "12px 16px",
    background: "rgba(248, 113, 113, 0.1)",
    border: "1px solid var(--accent-red)",
    borderRadius: "8px",
    color: "var(--accent-red)",
    fontSize: "13px",
    zIndex: 100,
  },
  loadingOverlay: {
    position: "absolute",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(9, 9, 11, 0.8)",
    color: "var(--text-muted)",
    fontSize: "14px",
    zIndex: 100,
  },
  spinner: {
    width: "32px",
    height: "32px",
    border: "3px solid var(--border-color)",
    borderTopColor: "var(--accent-primary)",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
    marginBottom: "12px",
  },
  searchResults: {
    marginTop: "8px",
    maxHeight: "200px",
    overflowY: "auto",
  },
  searchResult: {
    padding: "8px 10px",
    background: "var(--bg-tertiary)",
    borderRadius: "6px",
    marginBottom: "6px",
    cursor: "pointer",
  },
  searchResultHandle: {
    fontSize: "13px",
    fontWeight: 500,
    color: "var(--accent-blue)",
  },
  searchResultName: {
    fontSize: "11px",
    color: "var(--text-muted)",
  },
  nodePanel: {
    position: "absolute",
    top: "16px",
    right: "16px",
    width: "300px",
    background: "var(--bg-secondary)",
    border: "1px solid var(--border-color)",
    borderRadius: "12px",
    padding: "16px",
    zIndex: 100,
  },
  nodePanelHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: "12px",
  },
  nodeHandle: {
    fontSize: "15px",
    fontWeight: 600,
    color: "var(--accent-blue)",
  },
  nodeName: {
    fontSize: "12px",
    color: "var(--text-muted)",
  },
  closeButton: {
    background: "none",
    border: "none",
    color: "var(--text-muted)",
    fontSize: "20px",
    cursor: "pointer",
    padding: 0,
    lineHeight: 1,
  },
  nodeBio: {
    fontSize: "13px",
    color: "var(--text-secondary)",
    lineHeight: 1.5,
    marginBottom: "12px",
    maxHeight: "80px",
    overflow: "hidden",
  },
  nodeTags: {
    display: "flex",
    flexWrap: "wrap",
    gap: "6px",
    marginBottom: "12px",
  },
  nodeTag: {
    padding: "3px 8px",
    background: "var(--bg-tertiary)",
    borderRadius: "4px",
    fontSize: "10px",
    color: "var(--text-secondary)",
    textTransform: "uppercase",
  },
  tagSeed: {
    background: "rgba(248, 113, 113, 0.2)",
    color: "#f87171",
  },
  tagPending: {
    background: "rgba(251, 191, 36, 0.2)",
    color: "#fbbf24",
  },
  tagRelevant: {
    background: "rgba(52, 211, 153, 0.2)",
    color: "#34d399",
  },
  nodeStats: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: "8px",
    marginBottom: "12px",
  },
  nodeStat: {
    background: "var(--bg-tertiary)",
    borderRadius: "6px",
    padding: "10px",
  },
  nodeStatValue: {
    fontSize: "16px",
    fontWeight: 600,
    color: "var(--text-primary)",
  },
  nodeStatLabel: {
    fontSize: "10px",
    color: "var(--text-muted)",
    textTransform: "uppercase",
  },
  viewOnX: {
    display: "block",
    width: "100%",
    padding: "10px",
    background: "var(--accent-primary)",
    color: "#000",
    textAlign: "center",
    borderRadius: "8px",
    fontSize: "13px",
    fontWeight: 500,
    textDecoration: "none",
  },
  grokReasonSection: {
    marginBottom: "12px",
    padding: "10px",
    background: "var(--bg-tertiary)",
    borderRadius: "8px",
    borderLeft: "3px solid var(--accent-green)",
  },
  grokReasonLabel: {
    fontSize: "10px",
    fontWeight: 600,
    color: "var(--accent-green)",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: "6px",
  },
  grokReasonText: {
    fontSize: "12px",
    color: "var(--text-secondary)",
    lineHeight: 1.5,
  },
  loadingDetails: {
    fontSize: "12px",
    color: "var(--text-muted)",
    marginBottom: "12px",
    fontStyle: "italic",
  },
  floatingLegend: {
    position: "absolute",
    bottom: "20px",
    left: "20px",
    background: "rgba(24, 24, 27, 0.9)",
    backdropFilter: "blur(8px)",
    border: "1px solid rgba(63, 63, 70, 0.5)",
    borderRadius: "12px",
    padding: "16px",
    zIndex: 50,
  },
  floatingLegendTitle: {
    fontSize: "11px",
    fontWeight: 600,
    color: "#a1a1aa",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: "12px",
  },
  floatingLegendItems: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  floatingLegendItem: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    fontSize: "13px",
    color: "#e4e4e7",
  },
  floatingLegendDot: {
    width: "12px",
    height: "12px",
    borderRadius: "50%",
    flexShrink: 0,
  },
  floatingLegendNote: {
    fontSize: "11px",
    color: "#71717a",
    marginTop: "10px",
    paddingTop: "10px",
    borderTop: "1px solid rgba(63, 63, 70, 0.5)",
  },
};

export default GraphView;
