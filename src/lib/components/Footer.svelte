<script lang="ts">
	import { site, socials } from '$lib/site';
	import SocialLinks from './SocialLinks.svelte';
</script>

<footer class="mt-20 border-t border-border bg-surface/60">
	<div class="wide-container flex flex-col gap-5 py-8 text-sm text-muted sm:flex-row sm:items-center sm:justify-between">
		<div class="max-w-xl leading-6">
			<p>© {new Date().getFullYear()} {site.name}.</p>
			<p>{@html site.footerText}</p>
		</div>
		<SocialLinks links={socials} />
	</div>
</footer>
