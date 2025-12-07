import React from 'react';

interface AboutPageProps {
  onClose: () => void;
}

export function AboutPage({ onClose }: AboutPageProps) {
  return (
    <div style={styles.overlay}>
      <div style={styles.container}>
        <button style={styles.closeButton} onClick={onClose}>
          &times;
        </button>

        <div style={styles.header}>
          <h1 style={styles.title}>Grok Underrated Recruiter</h1>
          <p style={styles.tagline}>Finding Hidden Gems in the Taste Graph</p>
        </div>

        <div style={styles.content}>
          {/* The Problem */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>The Problem</h2>
            <div style={styles.quote}>
              "Elon personally interviewed the first 3,000 SpaceX hires. He finds exceptional
              people in weird corners of the internet — engineers with 800 followers doing
              incredible work that LinkedIn will never surface."
            </div>
            <p style={styles.text}>
              Traditional recruiting optimizes for <strong>polish over substance</strong>.
              Resume keywords, LinkedIn endorsements, and credential signaling miss the
              builders who are too busy shipping to self-promote.
            </p>
          </section>

          {/* Our Approach */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Our Approach: Taste Graphs</h2>
            <p style={styles.text}>
              We reverse-engineer <strong>Elon's implicit taste function</strong> by analyzing
              who xAI engineers interact with on X. Likes, retweets, and substantive replies
              reveal who the team respects — before they're famous.
            </p>
            <div style={styles.diagram}>
              <div style={styles.diagramRow}>
                <div style={styles.diagramBox}>
                  <div style={styles.diagramIcon}>🌱</div>
                  <div style={styles.diagramLabel}>Seeds</div>
                  <div style={styles.diagramDesc}>Elon + xAI Engineers</div>
                </div>
                <div style={styles.arrow}>→</div>
                <div style={styles.diagramBox}>
                  <div style={styles.diagramIcon}>🕸️</div>
                  <div style={styles.diagramLabel}>Taste Graph</div>
                  <div style={styles.diagramDesc}>16k+ Candidates</div>
                </div>
                <div style={styles.arrow}>→</div>
                <div style={styles.diagramBox}>
                  <div style={styles.diagramIcon}>🎯</div>
                  <div style={styles.diagramLabel}>Hidden Gems</div>
                  <div style={styles.diagramDesc}>Top 200 Ranked</div>
                </div>
              </div>
            </div>
          </section>

          {/* Pipeline */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>The Pipeline</h2>
            <div style={styles.pipeline}>
              <div style={styles.pipelineStep}>
                <div style={styles.stepNumber}>1</div>
                <div style={styles.stepContent}>
                  <h3 style={styles.stepTitle}>Graph Expansion</h3>
                  <p style={styles.stepDesc}>
                    Start with seed accounts. Expand via likes, retweets, replies.
                    Build interaction edges with weighted signals.
                  </p>
                </div>
              </div>

              <div style={styles.pipelineStep}>
                <div style={styles.stepNumber}>2</div>
                <div style={styles.stepContent}>
                  <h3 style={styles.stepTitle}>Fast LLM Screen</h3>
                  <p style={styles.stepDesc}>
                    <code>grok-4-1-fast</code> screens bios for technical relevance.
                    Filters 16k → 500 candidates in minutes.
                  </p>
                </div>
              </div>

              <div style={styles.pipelineStep}>
                <div style={styles.stepNumber}>3</div>
                <div style={styles.stepContent}>
                  <h3 style={styles.stepTitle}>PageRank + Underratedness</h3>
                  <p style={styles.stepDesc}>
                    Personalized PageRank with seeds as the personalization vector.
                    <strong> Underratedness = PPR / log(followers)</strong> finds hidden gems.
                  </p>
                </div>
              </div>

              <div style={styles.pipelineStep}>
                <div style={styles.stepNumber}>4</div>
                <div style={styles.stepContent}>
                  <h3 style={styles.stepTitle}>Deep Evaluation</h3>
                  <p style={styles.stepDesc}>
                    <code>grok-4-1-reasoning</code> + xAI Search Tools analyzes GitHub,
                    LinkedIn, and X posts. Scores on 5-criterion rubric.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Scoring Rubric */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Scoring Rubric</h2>
            <p style={styles.text}>
              Each candidate is scored 0-100 based on what Musk looks for in interviews:
            </p>
            <div style={styles.rubricGrid}>
              <div style={styles.rubricItem}>
                <div style={styles.rubricHeader}>
                  <span style={styles.rubricWeight}>25%</span>
                  <span style={styles.rubricName}>Technical Depth</span>
                </div>
                <p style={styles.rubricDesc}>
                  Systems knowledge, research contributions, not just buzzwords
                </p>
              </div>

              <div style={styles.rubricItem}>
                <div style={styles.rubricHeader}>
                  <span style={styles.rubricWeight}>25%</span>
                  <span style={styles.rubricName}>Project Evidence</span>
                </div>
                <p style={styles.rubricDesc}>
                  GitHub repos, shipped products, open source contributions
                </p>
              </div>

              <div style={styles.rubricItem}>
                <div style={styles.rubricHeader}>
                  <span style={styles.rubricWeight}>20%</span>
                  <span style={styles.rubricName}>Mission Alignment</span>
                </div>
                <p style={styles.rubricDesc}>
                  Obsessed with AI/AGI, posts about frontier research
                </p>
              </div>

              <div style={styles.rubricItem}>
                <div style={styles.rubricHeader}>
                  <span style={styles.rubricWeight}>20%</span>
                  <span style={styles.rubricName}>Exceptional Ability</span>
                </div>
                <p style={styles.rubricDesc}>
                  Evidence of solving hard problems, clear ownership
                </p>
              </div>

              <div style={styles.rubricItem}>
                <div style={styles.rubricHeader}>
                  <span style={styles.rubricWeight}>10%</span>
                  <span style={styles.rubricName}>Communication</span>
                </div>
                <p style={styles.rubricDesc}>
                  Technical writing quality, explains complex ideas clearly
                </p>
              </div>
            </div>
          </section>

          {/* Why This Works */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Why This Works</h2>
            <div style={styles.whyGrid}>
              <div style={styles.whyItem}>
                <div style={styles.whyIcon}>🔍</div>
                <h3 style={styles.whyTitle}>Signal over Noise</h3>
                <p style={styles.whyDesc}>
                  A like from an xAI engineer means more than 10k LinkedIn connections
                </p>
              </div>
              <div style={styles.whyItem}>
                <div style={styles.whyIcon}>💎</div>
                <h3 style={styles.whyTitle}>Anti-LinkedIn</h3>
                <p style={styles.whyDesc}>
                  Optimizes for substance over polish — finds builders, not self-promoters
                </p>
              </div>
              <div style={styles.whyItem}>
                <div style={styles.whyIcon}>🧠</div>
                <h3 style={styles.whyTitle}>Grok-Native</h3>
                <p style={styles.whyDesc}>
                  Uses xAI's own models + Search Tools for deep candidate analysis
                </p>
              </div>
              <div style={styles.whyItem}>
                <div style={styles.whyIcon}>📊</div>
                <h3 style={styles.whyTitle}>Explainable</h3>
                <p style={styles.whyDesc}>
                  Every score comes with evidence and reasoning — not a black box
                </p>
              </div>
            </div>
          </section>

          {/* Tech Stack */}
          <section style={styles.section}>
            <h2 style={styles.sectionTitle}>Built With</h2>
            <div style={styles.techStack}>
              <span style={styles.techBadge}>Grok 4.1</span>
              <span style={styles.techBadge}>xAI Search Tools</span>
              <span style={styles.techBadge}>X API v2</span>
              <span style={styles.techBadge}>NetworkX PageRank</span>
              <span style={styles.techBadge}>FastAPI</span>
              <span style={styles.techBadge}>React</span>
            </div>
          </section>

          {/* Footer */}
          <div style={styles.footer}>
            <p>Built for the xAI Hackathon 2024</p>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
    padding: '40px 20px',
    overflowY: 'auto',
    zIndex: 1000,
  },
  container: {
    backgroundColor: 'var(--bg-primary)',
    borderRadius: '16px',
    maxWidth: '800px',
    width: '100%',
    position: 'relative',
    border: '1px solid var(--border-color)',
  },
  closeButton: {
    position: 'absolute',
    top: '16px',
    right: '16px',
    background: 'none',
    border: 'none',
    fontSize: '28px',
    color: 'var(--text-secondary)',
    cursor: 'pointer',
    padding: '4px 12px',
    borderRadius: '8px',
  },
  header: {
    padding: '48px 48px 24px',
    textAlign: 'center',
    borderBottom: '1px solid var(--border-color)',
  },
  title: {
    fontSize: '36px',
    fontWeight: 700,
    margin: 0,
    background: 'linear-gradient(135deg, #10b981 0%, #3b82f6 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  tagline: {
    fontSize: '18px',
    color: 'var(--text-secondary)',
    marginTop: '8px',
  },
  content: {
    padding: '32px 48px 48px',
  },
  section: {
    marginBottom: '40px',
  },
  sectionTitle: {
    fontSize: '24px',
    fontWeight: 600,
    marginBottom: '16px',
    color: 'var(--text-primary)',
  },
  quote: {
    padding: '20px 24px',
    backgroundColor: 'var(--bg-secondary)',
    borderLeft: '4px solid #10b981',
    borderRadius: '0 8px 8px 0',
    fontStyle: 'italic',
    color: 'var(--text-secondary)',
    marginBottom: '16px',
    lineHeight: 1.6,
  },
  text: {
    fontSize: '16px',
    lineHeight: 1.7,
    color: 'var(--text-secondary)',
  },
  diagram: {
    padding: '32px',
    backgroundColor: 'var(--bg-secondary)',
    borderRadius: '12px',
    marginTop: '20px',
  },
  diagramRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '16px',
    flexWrap: 'wrap',
  },
  diagramBox: {
    textAlign: 'center',
    padding: '20px 24px',
    backgroundColor: 'var(--bg-primary)',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
    minWidth: '140px',
  },
  diagramIcon: {
    fontSize: '32px',
    marginBottom: '8px',
  },
  diagramLabel: {
    fontWeight: 600,
    fontSize: '14px',
    color: 'var(--text-primary)',
  },
  diagramDesc: {
    fontSize: '12px',
    color: 'var(--text-muted)',
    marginTop: '4px',
  },
  arrow: {
    fontSize: '24px',
    color: 'var(--text-muted)',
  },
  pipeline: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
  },
  pipelineStep: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-start',
  },
  stepNumber: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: '#10b981',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 700,
    fontSize: '16px',
    flexShrink: 0,
  },
  stepContent: {
    flex: 1,
    paddingTop: '4px',
  },
  stepTitle: {
    fontSize: '16px',
    fontWeight: 600,
    margin: '0 0 6px 0',
    color: 'var(--text-primary)',
  },
  stepDesc: {
    fontSize: '14px',
    color: 'var(--text-secondary)',
    margin: 0,
    lineHeight: 1.5,
  },
  rubricGrid: {
    display: 'grid',
    gap: '12px',
    marginTop: '16px',
  },
  rubricItem: {
    padding: '16px 20px',
    backgroundColor: 'var(--bg-secondary)',
    borderRadius: '10px',
    border: '1px solid var(--border-color)',
  },
  rubricHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '8px',
  },
  rubricWeight: {
    backgroundColor: '#10b981',
    color: 'white',
    padding: '4px 10px',
    borderRadius: '20px',
    fontSize: '12px',
    fontWeight: 600,
  },
  rubricName: {
    fontWeight: 600,
    fontSize: '15px',
    color: 'var(--text-primary)',
  },
  rubricDesc: {
    margin: 0,
    fontSize: '14px',
    color: 'var(--text-secondary)',
  },
  whyGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '16px',
    marginTop: '16px',
  },
  whyItem: {
    padding: '20px',
    backgroundColor: 'var(--bg-secondary)',
    borderRadius: '12px',
    border: '1px solid var(--border-color)',
  },
  whyIcon: {
    fontSize: '28px',
    marginBottom: '12px',
  },
  whyTitle: {
    fontSize: '16px',
    fontWeight: 600,
    margin: '0 0 8px 0',
    color: 'var(--text-primary)',
  },
  whyDesc: {
    margin: 0,
    fontSize: '14px',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
  },
  techStack: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '10px',
    marginTop: '12px',
  },
  techBadge: {
    padding: '8px 16px',
    backgroundColor: 'var(--bg-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: '20px',
    fontSize: '14px',
    color: 'var(--text-secondary)',
  },
  footer: {
    textAlign: 'center',
    paddingTop: '32px',
    borderTop: '1px solid var(--border-color)',
    color: 'var(--text-muted)',
    fontSize: '14px',
  },
};
