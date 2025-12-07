-- Grok Underrated Recruiter - Initial Schema
-- Run this in Supabase SQL Editor

-- =====================================================
-- CORE TABLES
-- =====================================================

-- Users table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    x_user_id TEXT UNIQUE,
    x_handle TEXT,
    x_name TEXT,
    x_profile_image_url TEXT,
    x_access_token TEXT,            -- Encrypted OAuth token
    x_access_token_secret TEXT,     -- Encrypted OAuth secret
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    settings JSONB DEFAULT '{}'
);

-- Graphs - supports shared base graph + user-specific graphs
CREATE TABLE IF NOT EXISTS public.graphs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    is_base_graph BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    last_updated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    settings JSONB DEFAULT '{}'
);

-- Seeds (root accounts for graph building)
CREATE TABLE IF NOT EXISTS public.seeds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES public.graphs(id) ON DELETE CASCADE,
    x_user_id TEXT NOT NULL,
    x_handle TEXT NOT NULL,
    x_name TEXT,
    x_bio TEXT,
    x_followers_count INTEGER DEFAULT 0,
    x_following_count INTEGER DEFAULT 0,
    added_by_user_id UUID REFERENCES public.users(id),
    is_active BOOLEAN DEFAULT TRUE,
    last_crawled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(graph_id, x_user_id)
);

-- Graph nodes (X accounts discovered)
CREATE TABLE IF NOT EXISTS public.nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES public.graphs(id) ON DELETE CASCADE,
    x_user_id TEXT NOT NULL,
    x_handle TEXT NOT NULL,
    x_name TEXT,
    x_bio TEXT,
    x_location TEXT,
    x_followers_count INTEGER DEFAULT 0,
    x_following_count INTEGER DEFAULT 0,
    x_tweet_count INTEGER DEFAULT 0,
    x_profile_image_url TEXT,

    -- Graph metadata
    is_seed BOOLEAN DEFAULT FALSE,
    is_candidate BOOLEAN DEFAULT FALSE,
    discovered_via TEXT,
    discovery_depth INTEGER DEFAULT 1,

    -- Ranking scores
    pagerank_score FLOAT DEFAULT 0,
    underratedness_score FLOAT DEFAULT 0,

    -- Evaluation status
    fast_screened BOOLEAN DEFAULT FALSE,
    fast_screen_passed BOOLEAN,
    fast_screen_reason TEXT,
    deep_evaluated BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(graph_id, x_user_id)
);

-- Graph edges (interactions between accounts)
CREATE TABLE IF NOT EXISTS public.edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id UUID NOT NULL REFERENCES public.graphs(id) ON DELETE CASCADE,
    src_node_id UUID NOT NULL REFERENCES public.nodes(id) ON DELETE CASCADE,
    dst_node_id UUID NOT NULL REFERENCES public.nodes(id) ON DELETE CASCADE,
    interaction_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    tweet_id TEXT,
    created_at_x TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    depth INTEGER DEFAULT 1
);

-- Deep evaluations (Grok analysis results)
CREATE TABLE IF NOT EXISTS public.evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES public.nodes(id) ON DELETE CASCADE,

    -- Scores (0-10)
    technical_depth_score INTEGER,
    technical_depth_evidence TEXT,
    project_evidence_score INTEGER,
    project_evidence_text TEXT,
    mission_alignment_score INTEGER,
    mission_alignment_evidence TEXT,
    exceptional_ability_score INTEGER,
    exceptional_ability_evidence TEXT,
    communication_score INTEGER,
    communication_evidence TEXT,

    -- Composite
    final_score FLOAT,

    -- Assessment
    summary TEXT,
    strengths TEXT[],
    concerns TEXT[],
    recommended_role TEXT,

    -- External links found
    github_url TEXT,
    linkedin_url TEXT,
    top_repos TEXT[],

    -- Metadata
    model_used TEXT,
    citations TEXT[],
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(node_id)
);

-- User's saved candidates (per-user bookmarks)
CREATE TABLE IF NOT EXISTS public.saved_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES public.nodes(id) ON DELETE CASCADE,
    notes TEXT,
    tags TEXT[],
    saved_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, node_id)
);

-- DM history (generated outreach messages)
CREATE TABLE IF NOT EXISTS public.dm_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    node_id UUID NOT NULL REFERENCES public.nodes(id) ON DELETE CASCADE,
    custom_context TEXT,
    tone TEXT DEFAULT 'professional',
    generated_message TEXT NOT NULL,
    character_count INTEGER,
    sent_via_x BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Background jobs
CREATE TABLE IF NOT EXISTS public.jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    graph_id UUID REFERENCES public.graphs(id) ON DELETE CASCADE,
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    payload JSONB DEFAULT '{}',
    result JSONB,
    error TEXT,
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    scheduled_for TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- INDEXES
-- =====================================================

CREATE INDEX IF NOT EXISTS idx_nodes_graph_candidate ON public.nodes(graph_id) WHERE is_candidate = TRUE;
CREATE INDEX IF NOT EXISTS idx_nodes_graph_handle ON public.nodes(graph_id, x_handle);
CREATE INDEX IF NOT EXISTS idx_nodes_underratedness ON public.nodes(graph_id, underratedness_score DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_pagerank ON public.nodes(graph_id, pagerank_score DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_deep_evaluated ON public.nodes(graph_id) WHERE deep_evaluated = TRUE;

CREATE INDEX IF NOT EXISTS idx_edges_graph ON public.edges(graph_id);
CREATE INDEX IF NOT EXISTS idx_edges_src ON public.edges(src_node_id);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON public.edges(dst_node_id);

CREATE INDEX IF NOT EXISTS idx_evaluations_score ON public.evaluations(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_role ON public.evaluations(recommended_role) WHERE final_score >= 50;

CREATE INDEX IF NOT EXISTS idx_jobs_status ON public.jobs(status, scheduled_for);
CREATE INDEX IF NOT EXISTS idx_jobs_graph ON public.jobs(graph_id, type);

CREATE INDEX IF NOT EXISTS idx_saved_user ON public.saved_candidates(user_id);
CREATE INDEX IF NOT EXISTS idx_dm_user ON public.dm_history(user_id);

-- =====================================================
-- ROW-LEVEL SECURITY POLICIES
-- =====================================================

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.graphs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.seeds ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.saved_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.dm_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

-- Users can read/update their own data
CREATE POLICY users_own ON public.users
    FOR ALL USING (auth.uid() = id);

-- Graphs: read base graph + own graphs + public graphs
CREATE POLICY graphs_read ON public.graphs
    FOR SELECT USING (
        is_base_graph = TRUE OR
        owner_id = auth.uid() OR
        is_public = TRUE
    );

CREATE POLICY graphs_write ON public.graphs
    FOR ALL USING (owner_id = auth.uid());

-- Seeds: read from accessible graphs, write to own graphs
CREATE POLICY seeds_read ON public.seeds
    FOR SELECT USING (
        graph_id IN (
            SELECT id FROM public.graphs
            WHERE is_base_graph = TRUE OR owner_id = auth.uid() OR is_public = TRUE
        )
    );

CREATE POLICY seeds_write ON public.seeds
    FOR INSERT WITH CHECK (
        graph_id IN (SELECT id FROM public.graphs WHERE owner_id = auth.uid())
    );

-- Nodes: read from accessible graphs
CREATE POLICY nodes_read ON public.nodes
    FOR SELECT USING (
        graph_id IN (
            SELECT id FROM public.graphs
            WHERE is_base_graph = TRUE OR owner_id = auth.uid() OR is_public = TRUE
        )
    );

-- Edges: read from accessible graphs
CREATE POLICY edges_read ON public.edges
    FOR SELECT USING (
        graph_id IN (
            SELECT id FROM public.graphs
            WHERE is_base_graph = TRUE OR owner_id = auth.uid() OR is_public = TRUE
        )
    );

-- Evaluations: read from accessible nodes
CREATE POLICY evaluations_read ON public.evaluations
    FOR SELECT USING (
        node_id IN (
            SELECT n.id FROM public.nodes n
            JOIN public.graphs g ON n.graph_id = g.id
            WHERE g.is_base_graph = TRUE OR g.owner_id = auth.uid() OR g.is_public = TRUE
        )
    );

-- Saved candidates: own only
CREATE POLICY saved_own ON public.saved_candidates
    FOR ALL USING (user_id = auth.uid());

-- DM history: own only
CREATE POLICY dm_own ON public.dm_history
    FOR ALL USING (user_id = auth.uid());

-- Jobs: own only
CREATE POLICY jobs_own ON public.jobs
    FOR ALL USING (user_id = auth.uid());

-- =====================================================
-- SERVICE ROLE POLICIES (for backend)
-- =====================================================

-- Allow service role to bypass RLS for all tables
-- This is done via Supabase dashboard or by using service_role key

-- =====================================================
-- FUNCTIONS
-- =====================================================

-- Function to get hybrid candidates (base graph + user's additional seeds)
CREATE OR REPLACE FUNCTION get_hybrid_candidates(
    p_user_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 30,
    p_offset INTEGER DEFAULT 0,
    p_sort_by TEXT DEFAULT 'final_score',
    p_min_score FLOAT DEFAULT 0,
    p_role TEXT DEFAULT NULL
)
RETURNS TABLE (
    node_id UUID,
    x_handle TEXT,
    x_name TEXT,
    x_bio TEXT,
    x_followers_count INTEGER,
    pagerank_score FLOAT,
    underratedness_score FLOAT,
    final_score FLOAT,
    recommended_role TEXT,
    summary TEXT,
    strengths TEXT[],
    github_url TEXT,
    linkedin_url TEXT,
    technical_depth_score INTEGER,
    technical_depth_evidence TEXT,
    project_evidence_score INTEGER,
    project_evidence_text TEXT,
    mission_alignment_score INTEGER,
    mission_alignment_evidence TEXT,
    exceptional_ability_score INTEGER,
    exceptional_ability_evidence TEXT,
    communication_score INTEGER,
    communication_evidence TEXT,
    is_from_user_graph BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH accessible_graphs AS (
        SELECT g.id,
               CASE WHEN g.owner_id = p_user_id THEN TRUE ELSE FALSE END AS is_user_graph
        FROM public.graphs g
        WHERE g.is_base_graph = TRUE
           OR g.owner_id = p_user_id
           OR g.is_public = TRUE
    )
    SELECT DISTINCT ON (n.x_user_id)
        n.id AS node_id,
        n.x_handle,
        n.x_name,
        n.x_bio,
        n.x_followers_count,
        n.pagerank_score,
        n.underratedness_score,
        COALESCE(e.final_score, 0::FLOAT) AS final_score,
        e.recommended_role,
        e.summary,
        e.strengths,
        e.github_url,
        e.linkedin_url,
        e.technical_depth_score,
        e.technical_depth_evidence,
        e.project_evidence_score,
        e.project_evidence_text,
        e.mission_alignment_score,
        e.mission_alignment_evidence,
        e.exceptional_ability_score,
        e.exceptional_ability_evidence,
        e.communication_score,
        e.communication_evidence,
        ag.is_user_graph AS is_from_user_graph
    FROM public.nodes n
    JOIN accessible_graphs ag ON n.graph_id = ag.id
    LEFT JOIN public.evaluations e ON n.id = e.node_id
    WHERE n.is_candidate = TRUE
      AND n.deep_evaluated = TRUE
      AND (p_min_score = 0 OR COALESCE(e.final_score, 0) >= p_min_score)
      AND (p_role IS NULL OR e.recommended_role = p_role)
    ORDER BY
        n.x_user_id,
        ag.is_user_graph DESC,
        CASE p_sort_by
            WHEN 'final_score' THEN COALESCE(e.final_score, 0)
            WHEN 'underratedness_score' THEN n.underratedness_score
            WHEN 'pagerank_score' THEN n.pagerank_score
            ELSE COALESCE(e.final_score, 0)
        END DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update graph node/edge counts
CREATE OR REPLACE FUNCTION update_graph_counts()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE public.graphs
    SET node_count = (SELECT COUNT(*) FROM public.nodes WHERE graph_id = NEW.graph_id),
        edge_count = (SELECT COUNT(*) FROM public.edges WHERE graph_id = NEW.graph_id),
        last_updated_at = NOW()
    WHERE id = NEW.graph_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updating counts
CREATE TRIGGER update_node_count
    AFTER INSERT OR DELETE ON public.nodes
    FOR EACH ROW EXECUTE FUNCTION update_graph_counts();

CREATE TRIGGER update_edge_count
    AFTER INSERT OR DELETE ON public.edges
    FOR EACH ROW EXECUTE FUNCTION update_graph_counts();

-- =====================================================
-- INITIAL DATA: Base Graph
-- =====================================================

-- Create the base graph (shared xAI seeds graph)
INSERT INTO public.graphs (name, is_base_graph, is_public)
VALUES ('xAI Seeds Graph', TRUE, TRUE)
ON CONFLICT DO NOTHING;
