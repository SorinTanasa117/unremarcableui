import { chromium, Browser, BrowserContext } from 'playwright';

let browser: Browser | null = null;
let context: BrowserContext | null = null;

export interface BrowseResult {
  status: number;
  url: string;
  text: string;
}

export interface SearchResult {
  title: string;
  url: string;
  description: string;
}

export async function getBrowser(): Promise<BrowserContext> {
  if (!browser || !browser.isConnected()) {
    browser = await chromium.launch({
      headless: true,
      args: [
        '--disable-blink-features=AutomationControlled',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-infobars',
        '--window-position=0,0',
        '--ignore-certificate-errors',
      ],
    });
  }
  if (!context) {
    context = await browser.newContext({
      userAgent:
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      viewport: { width: 1280, height: 720 },
      deviceScaleFactor: 1,
      locale: 'en-US',
      timezoneId: 'America/New_York',
    });
  }
  return context;
}

export async function browseUrl(url: string): Promise<BrowseResult> {
  const ctx = await getBrowser();
  const page = await ctx.newPage();
  try {
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20_000 });
    const status = response?.status() ?? 0;
    const text = await page.evaluate(() => {
      const doc = (globalThis as unknown as { document: any }).document;
      const removeTags = ['script', 'style', 'nav', 'footer', 'header', 'noscript', 'iframe'];
      removeTags.forEach((tag) => doc.querySelectorAll(tag).forEach((el: any) => el.remove()));
      return doc.body?.innerText ?? '';
    });
    return { status, url: page.url(), text: text.slice(0, 12_000) };
  } finally {
    await page.close();
  }
}

/**
 * Free browser fallback for public search result pages. It is rate-limited by
 * the caller and only used after the package-based DuckDuckGo request fails.
 */
export async function searchWithBrowser(query: string): Promise<SearchResult[]> {
  const ctx = await getBrowser();
  const page = await ctx.newPage();
  try {
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20_000 });
    const status = response?.status() ?? 0;
    if (status >= 400) throw new Error(`DuckDuckGo browser search returned HTTP ${status}.`);

    const results = await page.evaluate<SearchResult[]>(() => {
      const doc = (globalThis as unknown as { document: any }).document;
      return Array.from(doc.querySelectorAll('.result')).slice(0, 8).map((row: any) => {
      const link = row.querySelector('.result__a') as { textContent?: string; href?: string } | null;
      const description = row.querySelector('.result__snippet')?.textContent?.trim() ?? '';
      return { title: link?.textContent?.trim() ?? '', url: link?.href ?? '', description };
      }).filter((result: SearchResult) => result.title && result.url);
    });

    if (!results.length) throw new Error('DuckDuckGo browser search returned no usable results.');
    return results;
  } finally {
    await page.close();
  }
}

/** A separate free public search fallback, used only when DuckDuckGo is unavailable. */
export async function searchBingWithBrowser(query: string): Promise<SearchResult[]> {
  const ctx = await getBrowser();
  const page = await ctx.newPage();
  try {
    const url = `https://www.bing.com/search?q=${encodeURIComponent(query)}`;
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20_000 });
    const status = response?.status() ?? 0;
    if (status >= 400) throw new Error(`Bing browser search returned HTTP ${status}.`);

    // Handle Bing cookie consent banner if it appears
    try {
      const acceptBtn = page.locator('#bnp_btn_accept, #adlt_set_yes, .bnp_btn_accept');
      if (await acceptBtn.isVisible({ timeout: 2000 })) {
        await acceptBtn.click();
        await page.waitForTimeout(500);
      }
    } catch {}

    const results = await page.evaluate<SearchResult[]>(() => {
      const doc = (globalThis as unknown as { document: any }).document;
      return Array.from(doc.querySelectorAll('li.b_algo')).slice(0, 8).map((row: any) => {
        const link = row.querySelector('h2 a, .b_algo h2 a') as { textContent?: string; href?: string } | null;
        const descriptionEl = row.querySelector('.b_caption p, .b_algoSubspace, .b_linelimit2, .b_caption');
        const description = descriptionEl?.textContent?.trim() ?? '';
        return { title: link?.textContent?.trim() ?? '', url: link?.href ?? '', description };
      }).filter((result: SearchResult) => result.title && result.url);
    });

    if (!results.length) throw new Error('Bing browser search returned no usable results.');
    return results;
  } finally {
    await page.close();
  }
}

/** A fourth fallback search option using Yahoo Search, which has lightweight bot checks. */
export async function searchYahooWithBrowser(query: string): Promise<SearchResult[]> {
  const ctx = await getBrowser();
  const page = await ctx.newPage();
  try {
    const url = `https://search.yahoo.com/search?q=${encodeURIComponent(query)}`;
    const response = await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 20_000 });
    const status = response?.status() ?? 0;
    if (status >= 400) throw new Error(`Yahoo browser search returned HTTP ${status}.`);

    // Click cookie consent button if Yahoo displays one
    try {
      const consentBtn = page.locator('button[name="agree"], button.agree, #consent-btn');
      if (await consentBtn.isVisible({ timeout: 2000 })) {
        await consentBtn.click();
        await page.waitForTimeout(500);
      }
    } catch {}

    const results = await page.evaluate<SearchResult[]>(() => {
      const doc = (globalThis as unknown as { document: any }).document;
      return Array.from(doc.querySelectorAll('div.algo, .algo-sr, .dd.algo')).slice(0, 8).map((row: any) => {
        const link = row.querySelector('h3.title a, h3 a, a') as { textContent?: string; href?: string } | null;
        const descriptionEl = row.querySelector('div.compText, div.compText p, .compText, span.fc-lh-lg');
        const description = descriptionEl?.textContent?.trim() ?? '';
        return { title: link?.textContent?.trim() ?? '', url: link?.href ?? '', description };
      }).filter((result: SearchResult) => result.title && result.url);
    });

    if (!results.length) throw new Error('Yahoo browser search returned no usable results.');
    return results;
  } finally {
    await page.close();
  }
}

export async function closeBrowser() {
  await context?.close();
  await browser?.close();
  browser = null;
  context = null;
}
