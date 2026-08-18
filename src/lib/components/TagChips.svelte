<script lang="ts">
	let { tags, linked = true }: { tags: string[]; linked?: boolean } = $props();
</script>

{#if tags.length > 0}
	<div class="flex flex-wrap gap-2">
		{#each tags as tag (tag)}
			{#if linked}
				<a
					class="rounded-full bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent no-underline hover:bg-accent/20"
					href={`/blog/tag/${encodeURIComponent(tag)}/`}
				>
					#{tag}
				</a>
			{:else}
				<span class="rounded-full bg-accent/10 px-2.5 py-1 text-xs font-medium text-accent">#{tag}</span>
			{/if}
		{/each}
	</div>
{/if}
