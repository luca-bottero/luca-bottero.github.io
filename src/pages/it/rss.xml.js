import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = (await getCollection('blog', ({ data }) => data.lang === 'it'))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'Luca Bottero — Blog',
    description: 'Saggi e note tecniche: intelligenza artificiale, fisica, sistemi autonomi e ciò che le macchine pensanti significano per noi.',
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: `/it/blog/${post.data.postSlug}/`,
    })),
    customData: '<language>it</language>',
  });
}
