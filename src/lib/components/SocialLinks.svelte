<script lang="ts">
	import Github from '~icons/simple-icons/github';
	import Linkedin from '~icons/simple-icons/linkedin';
	import Rss from '~icons/lucide/rss';
	import Scholar from '~icons/academicons/google-scholar';
	import X from '~icons/simple-icons/x';
	import type { SocialLink } from '$lib/site';

	let { links }: { links: SocialLink[] } = $props();
</script>

<div class="flex flex-wrap items-center gap-3">
	{#each links as link (link.url)}
		<a
			class="inline-flex size-10 items-center justify-center rounded-full border border-border text-ink no-underline transition hover:border-accent hover:text-accent"
			href={link.url}
			aria-label={link.name}
			title={link.name}
			target="_blank"
			rel="noreferrer"
		>
			{#if link.icon === 'github'}
				<Github width={19} height={19} />
			{:else if link.icon === 'linkedin'}
				<Linkedin width={19} height={19} />
			{:else if link.icon === 'scholar'}
				<Scholar width="20" height="20" />
			{:else if link.icon === 'x'}
				<X width={17} height={17} />
			{:else}
				<Rss width={19} height={19} />
			{/if}
		</a>
	{/each}
</div>
