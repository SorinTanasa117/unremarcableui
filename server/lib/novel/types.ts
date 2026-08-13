/**
 * Novel Mode type definitions.
 * Mirrors the schema from the implementation plan.
 */

export interface NovelManifest {
  id: string;
  title: string;
  premise: string;
  created_at: string;
  updated_at: string;
  status: 'outlining' | 'drafting' | 'complete';
  model_id: string;
  persona: string;
  target_chapter_count: number;
  current_chapter: number;
  num_predict_novel: number;
  // Length targets
  target_pages: number;
  words_per_page: number;
  target_words: number;
  words_per_chapter: number;
  dedicated_chat?: boolean;
  outline_revision_used?: boolean;
  draft_outline_is_revision?: boolean;
}

export interface ChapterOutline {
  number: number;
  title: string;
  beat_summary: string;
  description?: string;
  pov_character: string;
  characters_involved?: string[];
  characters?: string[];
  target_pages?: number;
  summary?: string;
  pov?: string;
  target_words?: number;
  status: 'planned' | 'drafted' | 'revised';
}

export interface ChapterSummary {
  chapter: number;
  summary: string;
  characters_active: string[];
  open_threads: string[];
  tone_notes: string;
  last_600_words_raw: string;
}

export interface ArcSummary {
  arc_number: number;
  chapters_covered: number[];
  summary: string;
  characters_state: string[];
  major_threads: string[];
}

export interface NovelBible {
  pov?: string;
  characters: BibleCharacter[];
  locations: BibleLocation[];
  facts: string[];
  style_notes: string;
  steering_notes?: string[];
}

export interface BibleCharacter {
  name: string;
  description: string;
  traits: string[];
  relationships: string[];
}

export interface BibleLocation {
  name: string;
  description: string;
}

export interface NovelWithDetails {
  manifest: NovelManifest;
  outline: ChapterOutline[];
  draft_outline: ChapterOutline[];
  bible: NovelBible;
  chapters: { number: number; wordCount: number; status: string }[];
}

export interface DraftSegment {
  index: number;
  content: string;
  tokenCount: number;
}

export interface DraftResult {
  chapterNumber: number;
  segments: DraftSegment[];
  fullText: string;
  completed: boolean;
}
