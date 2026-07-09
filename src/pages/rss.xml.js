import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';

export async function GET(context) {
  const posts = (await getCollection('blog', ({ data }) => data.lang === 'en'))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
  return rss({
    title: 'Luca Bottero — Blog',
    description: 'Essays and technical notes: artificial intelligence, physics, autonomous systems, and what thinking machines mean for us.',
    site: context.site,
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.date,
      link: `/blog/${post.data.postSlug}/`,
    })),
    customData: '<language>en</language>',
  });
}
