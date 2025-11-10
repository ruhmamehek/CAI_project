<template>
  <section class="results-section">
    <div class="answer-box">
      <h2>Answer</h2>
      <div class="answer-content">{{ result.answer }}</div>
    </div>

    <div class="sources-box">
      <h2>Sources ({{ result.num_chunks_retrieved }})</h2>
      <div class="sources-list">
        <div
          v-for="(source, index) in result.sources"
          :key="index"
          class="source-item"
        >
          <div class="source-header">
            <span class="source-ticker">{{ source.ticker }}</span>
            <span class="source-filing">{{ source.filing_type }}</span>
            <span class="source-year">{{ source.year }}</span>
            <span class="source-score">Score: {{ source.score.toFixed(3) }}</span>
          </div>
          <div class="source-details">
            <small>Accession: {{ source.accession_number }}</small>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'Results',
  props: {
    result: {
      type: Object,
      required: true
    }
  }
}
</script>

