export type Lang = 'en' | 'it';

export const ui = {
  en: {
    nav: {
      nousynth: 'NouSynth',
      systems: 'Systems',
      science: 'Science',
      lab: 'Lab',
      record: 'Record',
      blog: 'Writing',
      about: 'About',
    },
    footer: {
      explore: 'Explore',
      connect: 'Connect',
      brand: 'NouSynth',
      papers: 'Independent papers',
      rights: 'Luca Bottero — built from first principles.',
      langName: 'Italiano',
    },
    misc: {
      readMore: 'Read more',
      allSystems: 'All systems',
      caseStudy: 'Case study',
      backTo: 'Back to',
      status: 'STATUS: BUILDING',
      location: '45.0703°N 7.6869°E — TURIN, ITALY',
      menu: 'MENU',
    },
  },
  it: {
    nav: {
      nousynth: 'NouSynth',
      systems: 'Sistemi',
      science: 'Scienza',
      lab: 'Lab',
      record: 'Percorso',
      blog: 'Scritti',
      about: 'Chi sono',
    },
    footer: {
      explore: 'Esplora',
      connect: 'Contatti',
      brand: 'NouSynth',
      papers: 'Paper indipendenti',
      rights: 'Luca Bottero — costruito dai principi primi.',
      langName: 'English',
    },
    misc: {
      readMore: 'Approfondisci',
      allSystems: 'Tutti i sistemi',
      caseStudy: 'Caso di studio',
      backTo: 'Torna a',
      status: 'STATO: IN COSTRUZIONE',
      location: '45.0703°N 7.6869°E — TORINO, ITALIA',
      menu: 'MENU',
    },
  },
} as const;

/** Map an EN path to its IT twin and vice versa. */
export function twinPath(lang: Lang, path: string): string {
  if (lang === 'en') return path === '/' ? '/it/' : `/it${path}`;
  const stripped = path.replace(/^\/it/, '');
  return stripped === '' ? '/' : stripped;
}

export function langPrefix(lang: Lang): string {
  return lang === 'it' ? '/it' : '';
}
