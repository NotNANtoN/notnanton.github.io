export const site = {
	firstName: 'Anton',
	middleName: 'Orell',
	lastName: 'Wiehe',
	get name() {
		return `${this.firstName} ${this.middleName} ${this.lastName}`;
	},
	get shortName() {
		return `${this.firstName} ${this.lastName}`;
	},
	description:
		'AI researcher and entrepreneur. Real-robot learning with imitation and reinforcement learning, and AI products shipped into regulated industries. Co-Founder of Pharos Labs and AdaLab.',
	keywords: [
		'robot-learning',
		'vision-language-action',
		'imitation-learning',
		'reinforcement-learning',
		'lerobot',
		'artificial-general-intelligence',
		'curiosity'
	],
	lang: 'en',
	url: 'https://notnanton.github.io',
	blogName: "Anton's Blog",
	blogDescription: 'Notes on robot learning, reinforcement learning, and building adaptive systems',
	contactNote: 'Reach me wherever :-)',
	footerText:
		'Built with <a href="https://svelte.dev/docs/kit">SvelteKit</a>. Hosted by <a href="https://pages.github.com/">GitHub Pages</a>.'
} as const;

export type SocialLink = {
	name: string;
	url: string;
	/** unplugin-icons component key, resolved in the Social component. */
	icon: 'github' | 'x' | 'linkedin' | 'scholar' | 'rss';
};

export const socials: SocialLink[] = [
	{ name: 'GitHub', url: 'https://github.com/NotNaNtoN', icon: 'github' },
	{
		name: 'Google Scholar',
		url: 'https://scholar.google.com/citations?user=XAgRA8sAAAAJ',
		icon: 'scholar'
	},
	{ name: 'LinkedIn', url: 'https://www.linkedin.com/in/antonwiehe', icon: 'linkedin' },
	{ name: 'X', url: 'https://x.com/antonwiehe', icon: 'x' }
];

export const nav = [
	{ title: 'Blog', href: '/blog/' },
	{ title: 'Publications', href: '/publications/' },
	{ title: 'Talks', href: '/talks/' },
	{ title: 'Projects', href: '/projects/' },
	{ title: 'CV', href: '/cv/' }
];

/**
 * Giscus comments. `categoryId` and `repoId` are required by giscus; while they are
 * empty the comment widget stays disabled instead of rendering a broken iframe.
 */
export const giscus = {
	repo: 'NotNANtoN/notnanton.github.io',
	repoId: '',
	category: 'Comments',
	categoryId: '',
	mapping: 'title',
	strict: '1',
	reactionsEnabled: '1',
	inputPosition: 'bottom' as const,
	get enabled() {
		return this.repoId !== '' && this.categoryId !== '';
	}
};

export const homeSections = {
	newsLimit: 5,
	latestPostsLimit: 3
};
