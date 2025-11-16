<template>
  <section class="results-section">
    <div class="answer-box">
      <h2>Answer</h2>
      <div class="answer-content" ref="answerContent" v-html="formattedAnswer"></div>
    </div>
    
    <!-- Custom Tooltip -->
    <div 
      v-if="tooltip.visible" 
      class="custom-tooltip"
      :style="{ top: tooltip.y + 'px', left: tooltip.x + 'px' }"
    >
      {{ tooltip.text }}
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
            <div class="source-header-left">
              <span class="source-ticker">{{ source.ticker }}</span>
              <span class="source-filing">{{ source.filing_type }}</span>
              <span class="source-year">{{ source.year }}</span>
              <span class="source-chunk-id">{{ source.chunk_id }}</span>
              <span class="source-score">Score: {{ source.score.toFixed(3) }}</span>
            </div>
            <button 
              v-if="source.text"
              @click="toggleSource(index)"
              class="source-expand-btn"
              :aria-expanded="isExpanded(index)"
              :title="isExpanded(index) ? 'Hide source text' : 'Show source text'"
            >
              <span class="expand-icon" :class="{ 'expanded': isExpanded(index) }">▼</span>
            </button>
          </div>
          <div class="source-details">
            <small>Accession: {{ source.accession_number }}</small>
          </div>
          <div 
            v-if="source.text && isExpanded(index)" 
            class="source-text-content"
          >
            <div class="source-text">{{ source.text }}</div>
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
  },
  data() {
    return {
      tooltip: {
        visible: false,
        text: '',
        x: 0,
        y: 0
      },
      expandedSources: new Set()  // Track which sources are expanded
    }
  },
  computed: {
    formattedAnswer() {
      if (!this.result?.answer) return ''
      
      // Regular expression to match <source> tags with all attributes
      // Matches: <source ticker="AAPL" filing_type="10-Q" year="2023" chunk_id="...">text</source>
      const sourceTagRegex = /<source\s+([^>]+)>([^<]+)<\/source>/gi
      
      let formatted = this.result.answer
      const parts = []
      let lastIndex = 0
      let match
      
      // Reset regex for iteration
      sourceTagRegex.lastIndex = 0
      
      // Process each match and build parts array
      while ((match = sourceTagRegex.exec(formatted)) !== null) {
        // Add text before the match (escape it)
        if (match.index > lastIndex) {
          parts.push({
            type: 'text',
            content: this.escapeHtml(formatted.substring(lastIndex, match.index))
          })
        }
        
        // Extract attributes from the source tag
        const attrs = {}
        const attrRegex = /(\w+)="([^"]+)"/g
        let attrMatch
        while ((attrMatch = attrRegex.exec(match[1])) !== null) {
          attrs[attrMatch[1]] = attrMatch[2]
        }
        
        // Add the highlighted span part
        parts.push({
          type: 'highlight',
          content: match[2], // The text inside the tag
          attrs: attrs
        })
        
        lastIndex = sourceTagRegex.lastIndex
      }
      
      // Add remaining text after last match
      if (lastIndex < formatted.length) {
        parts.push({
          type: 'text',
          content: this.escapeHtml(formatted.substring(lastIndex))
        })
      }
      
      // Build HTML string from parts
      return parts.map(part => {
        if (part.type === 'highlight') {
          const attrs = part.attrs
          const chunkId = attrs.chunk_id || 'Unknown'
          return `<span class="source-highlight" 
                        data-ticker="${this.escapeHtml(attrs.ticker || '')}" 
                        data-filing-type="${this.escapeHtml(attrs.filing_type || '')}" 
                        data-year="${this.escapeHtml(attrs.year || '')}" 
                        data-chunk-id="${this.escapeHtml(chunkId)}">${this.escapeHtml(part.content)}</span>`
        } else {
          return part.content
        }
      }).join('')
    }
  },
  methods: {
    escapeHtml(text) {
      const div = document.createElement('div')
      div.textContent = text
      return div.innerHTML
    },
    showTooltip(event, chunkId, ticker, filingType, year) {
      const tooltipText = `Chunk ID: ${chunkId} | Source: ${ticker} ${filingType} ${year}`
      this.tooltip = {
        visible: true,
        text: tooltipText,
        x: event.clientX + 10,
        y: event.clientY + 10
      }
    },
    hideTooltip() {
      this.tooltip.visible = false
    },
    toggleSource(index) {
      if (this.expandedSources.has(index)) {
        this.expandedSources.delete(index)
      } else {
        this.expandedSources.add(index)
      }
    },
    isExpanded(index) {
      return this.expandedSources.has(index)
    },
    attachTooltipListeners() {
      if (!this.$refs.answerContent) return
      
      const highlights = this.$refs.answerContent.querySelectorAll('.source-highlight')
      highlights.forEach(span => {
        // Remove existing listeners to avoid duplicates
        const newSpan = span.cloneNode(true)
        span.parentNode.replaceChild(newSpan, span)
        
        // Get data attributes
        const chunkId = newSpan.getAttribute('data-chunk-id') || 'Unknown'
        const ticker = newSpan.getAttribute('data-ticker') || 'Unknown'
        const filingType = newSpan.getAttribute('data-filing-type') || ''
        const year = newSpan.getAttribute('data-year') || ''
        
        // Attach event listeners
        newSpan.addEventListener('mouseenter', (e) => {
          this.showTooltip(e, chunkId, ticker, filingType, year)
        })
        newSpan.addEventListener('mouseleave', () => {
          this.hideTooltip()
        })
        newSpan.addEventListener('mousemove', (e) => {
          if (this.tooltip.visible) {
            this.tooltip.x = e.clientX + 10
            this.tooltip.y = e.clientY + 10
          }
        })
      })
    }
  },
  mounted() {
    this.$nextTick(() => {
      this.attachTooltipListeners()
    })
  },
  updated() {
    this.$nextTick(() => {
      this.attachTooltipListeners()
    })
  }
}
</script>

