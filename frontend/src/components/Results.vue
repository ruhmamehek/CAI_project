<template>
  <section class="results-section">
    <!-- Filter Selection Reasoning Section -->
    <div v-if="result.filter_reasoning" class="filter-reasoning-box">
      <div class="filter-reasoning-header">
        <h2>
          <span class="filter-icon">🔍</span>
          Document Filter Selection
        </h2>
        <div v-if="result.applied_filters" class="applied-filters">
          <span 
            v-for="(value, key) in result.applied_filters" 
            :key="key"
            class="filter-badge"
          >
            {{ key }}: {{ value }}
          </span>
        </div>
        <button 
          @click="toggleFilterReasoning" 
          class="toggle-filter-reasoning-btn"
          :aria-expanded="showFilterReasoning"
          :title="showFilterReasoning ? 'Hide filter reasoning' : 'Show filter reasoning'"
        >
          <span class="expand-icon" :class="{ 'expanded': showFilterReasoning }">▼</span>
        </button>
      </div>
      <div v-if="showFilterReasoning" class="filter-reasoning-content">
        <div class="filter-reasoning-text" v-html="formattedFilterReasoning"></div>
      </div>
    </div>

    <!-- Reasoning Steps Section -->
    <div v-if="result.reasoning_steps" class="reasoning-box">
      <div class="reasoning-header">
        <h2>
          <span class="reasoning-icon">🧠</span>
          Reasoning Process
        </h2>
        <button 
          @click="toggleReasoning" 
          class="toggle-reasoning-btn"
          :aria-expanded="showReasoning"
          :title="showReasoning ? 'Hide reasoning' : 'Show reasoning'"
        >
          <span class="expand-icon" :class="{ 'expanded': showReasoning }">▼</span>
        </button>
      </div>
      <div v-if="showReasoning" class="reasoning-content">
        <div class="reasoning-steps" v-html="formattedReasoning"></div>
      </div>
    </div>

    <div class="answer-box">
      <h2>Answer</h2>
      <div class="answer-content" ref="answerContent" v-html="formattedAnswer"></div>
    </div>

    <!-- Verification Section -->
    <div v-if="result.verification" class="verification-box">
      <div class="verification-header">
        <h2>
          <span class="verification-icon">✓</span>
          Verification Results
        </h2>
        <button 
          @click="toggleVerification" 
          class="toggle-verification-btn"
          :aria-expanded="showVerification"
          :title="showVerification ? 'Hide verification' : 'Show verification'"
        >
          <span class="expand-icon" :class="{ 'expanded': showVerification }">▼</span>
        </button>
      </div>
      <div v-if="showVerification" class="verification-content">
        <!-- Overall Score -->
        <div class="verification-score">
          <div class="score-label">Overall Verification Score:</div>
          <div class="score-value" :class="getScoreClass(result.verification.overall_score)">
            {{ (result.verification.overall_score * 100).toFixed(1) }}%
          </div>
        </div>

        <!-- Verified Numbers -->
        <div v-if="result.verification.verified_numbers && result.verification.verified_numbers.length > 0" class="verified-numbers-section">
          <h3 class="section-title verified-title">
            <span class="check-icon">✓</span>
            Verified Numbers ({{ result.verification.verified_numbers.length }})
          </h3>
          <div class="numbers-list">
            <span 
              v-for="(number, index) in result.verification.verified_numbers" 
              :key="index"
              class="number-badge verified-number"
            >
              {{ formatNumber(number) }}
            </span>
          </div>
        </div>

        <!-- Unverified Claims -->
        <div v-if="result.verification.unverified_claims && result.verification.unverified_claims.length > 0" class="unverified-claims-section">
          <h3 class="section-title unverified-title">
            <span class="warning-icon">⚠</span>
            Unverified Claims ({{ result.verification.unverified_claims.length }})
          </h3>
          <ul class="claims-list">
            <li 
              v-for="(claim, index) in result.verification.unverified_claims" 
              :key="index"
              class="claim-item"
            >
              {{ claim }}
            </li>
          </ul>
        </div>

        <!-- Verification Details -->
        <div class="verification-details">
          <div class="detail-item">
            <span class="detail-label">Answer-Source Alignment:</span>
            <span class="detail-value">{{ (result.verification.answer_source_alignment * 100).toFixed(1) }}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Citation Coverage:</span>
            <span class="detail-value">{{ (result.verification.citation_coverage * 100).toFixed(1) }}%</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">Fact Verification Score:</span>
            <span class="detail-value">{{ (result.verification.fact_verification_score * 100).toFixed(1) }}%</span>
          </div>
        </div>

        <!-- Issues -->
        <div v-if="result.verification.issues && result.verification.issues.length > 0" class="issues-section">
          <h3 class="section-title issues-title">
            <span class="info-icon">ℹ</span>
            Issues
          </h3>
          <ul class="issues-list">
            <li 
              v-for="(issue, index) in result.verification.issues" 
              :key="index"
              class="issue-item"
            >
              {{ issue }}
            </li>
          </ul>
        </div>
      </div>
    </div>
    
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
              <span class="source-year">{{ source.year }}</span>
              <span v-if="source.item_number" class="source-item-number">Item {{ source.item_number }}</span>
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
            <small>Chunk ID: {{ source.chunk_id }}</small>
            <small v-if="source.chunk_type !== 'Text'"> | Type: {{ source.chunk_type }}</small>
            <small v-if="source.page"> | Page: {{ source.page }}</small>
          </div>
          <div v-if="source.image_path" class="source-image-container">
            <img 
              :src="getImageUrl(source.image_path)" 
              :alt="`${source.chunk_type} from ${source.ticker} ${source.year}`"
              class="source-image"
              @error="handleImageError"
            />
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
      expandedSources: new Set(),  // Track which sources are expanded
      showReasoning: true,  // Show reasoning by default
      showFilterReasoning: true,  // Show filter reasoning by default
      showVerification: true,  // Show verification by default
      apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000'
    }
  },
  computed: {
    formattedFilterReasoning() {
      if (!this.result?.filter_reasoning) return ''
      
      // Format filter reasoning - preserve line breaks and structure
      let formatted = this.escapeHtml(this.result.filter_reasoning)
      
      // Convert step numbers or bullet points to better formatting
      formatted = formatted.replace(/Step\s+(\d+):/gi, '<strong>Step $1:</strong>')
      formatted = formatted.replace(/Final\s+Step:/gi, '<strong>Final Step:</strong>')
      
      // Preserve line breaks
      formatted = formatted.replace(/\n/g, '<br>')
      
      return formatted
    },
    formattedReasoning() {
      if (!this.result?.reasoning_steps) return ''
      
      // Format reasoning steps - preserve line breaks and structure
      let formatted = this.escapeHtml(this.result.reasoning_steps)
      
      // Convert step numbers or bullet points to better formatting
      formatted = formatted.replace(/Step\s+(\d+):/gi, '<strong>Step $1:</strong>')
      formatted = formatted.replace(/Final\s+Step:/gi, '<strong>Final Step:</strong>')
      
      // Preserve line breaks
      formatted = formatted.replace(/\n/g, '<br>')
      
      return formatted
    },
    formattedAnswer() {
      if (!this.result?.answer) return ''
      
      // Regular expression to match <source> tags with all attributes
      // Matches: <source ticker="AAPL" year="2023" chunk_id="...">text</source>
      const sourceTagRegex = /<source\s+([^>]+)>([^<]+)<\/source>/gi
      
      let formatted = this.result.answer
      const parts = []
      let lastIndex = 0
      let match
      
      sourceTagRegex.lastIndex = 0
      
      while ((match = sourceTagRegex.exec(formatted)) !== null) {
        if (match.index > lastIndex) {
          parts.push({
            type: 'text',
            content: this.escapeHtml(formatted.substring(lastIndex, match.index))
          })
        }
        
        const attrs = {}
        const attrRegex = /(\w+)="([^"]+)"/g
        let attrMatch
        while ((attrMatch = attrRegex.exec(match[1])) !== null) {
          attrs[attrMatch[1]] = attrMatch[2]
        }
        
        parts.push({
          type: 'highlight',
          content: match[2],
          attrs: attrs
        })
        
        lastIndex = sourceTagRegex.lastIndex
      }
      
      if (lastIndex < formatted.length) {
        parts.push({
          type: 'text',
          content: this.escapeHtml(formatted.substring(lastIndex))
        })
      }
      
      return parts.map(part => {
        if (part.type === 'highlight') {
          const attrs = part.attrs
          const chunkId = attrs.chunk_id || 'Unknown'
          return `<span class="source-highlight" 
                        data-ticker="${this.escapeHtml(attrs.ticker || '')}" 
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
    showTooltip(event, chunkId, ticker, year) {
      const tooltipText = `Chunk ID: ${chunkId} | Source: ${ticker} ${year}`
      this.tooltip.visible = true
      this.tooltip.text = tooltipText
      this.tooltip.x = event.clientX + 10
      this.tooltip.y = event.clientY + 10
    },
    hideTooltip() {
      this.tooltip.visible = false
    },
    updateTooltipPosition(event) {
      // Only update position, don't change visibility to avoid re-renders
      if (this.tooltip.visible) {
        this.tooltip.x = event.clientX + 10
        this.tooltip.y = event.clientY + 10
      }
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
    toggleReasoning() {
      this.showReasoning = !this.showReasoning
    },
    toggleFilterReasoning() {
      this.showFilterReasoning = !this.showFilterReasoning
    },
    toggleVerification() {
      this.showVerification = !this.showVerification
    },
    formatNumber(number) {
      // Format number with commas and handle large numbers
      if (number >= 1e9) {
        return `$${(number / 1e9).toFixed(2)}B`
      } else if (number >= 1e6) {
        return `$${(number / 1e6).toFixed(2)}M`
      } else if (number >= 1e3) {
        return `$${(number / 1e3).toFixed(2)}K`
      } else {
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 0,
          maximumFractionDigits: 2
        }).format(number)
      }
    },
    getScoreClass(score) {
      if (score >= 0.8) return 'score-high'
      if (score >= 0.6) return 'score-medium'
      return 'score-low'
    },
    getImageUrl(imagePath) {
      if (!imagePath) return ''
      // Construct the full URL to the backend image endpoint
      // image_path should be relative to project root (e.g., "data/images/table_123.png")
      return `${this.apiUrl}/images/${encodeURIComponent(imagePath)}`
    },
    handleImageError(event) {
      // Hide the image on error
      event.target.style.display = 'none'
      console.warn('Failed to load image:', event.target.src)
    },
    attachTooltipListeners() {
      if (!this.$refs.answerContent) return
      
      // Use event delegation on the parent container instead of individual elements
      // This prevents flickering because we don't need to re-attach listeners
      const container = this.$refs.answerContent
      
      // Remove existing listener if any
      if (container._tooltipHandler) {
        container.removeEventListener('mouseover', container._tooltipHandler)
        container.removeEventListener('mouseout', container._tooltipHandler)
        container.removeEventListener('mousemove', container._tooltipHandler)
      }
      
      // Create a single handler for event delegation
      container._tooltipHandler = (e) => {
        const target = e.target.closest('.source-highlight')
        if (!target) {
          // If mouse leaves a highlight and doesn't enter another, hide tooltip
          if (e.type === 'mouseout' && !e.relatedTarget?.closest('.source-highlight')) {
            this.hideTooltip()
          }
          return
        }
        
        // Get data attributes from the target
        const chunkId = target.getAttribute('data-chunk-id') || 'Unknown'
        const ticker = target.getAttribute('data-ticker') || 'Unknown'
        const year = target.getAttribute('data-year') || ''
        
        if (e.type === 'mouseover') {
          this.showTooltip(e, chunkId, ticker, year)
        } else if (e.type === 'mouseout') {
          // Only hide if we're leaving the highlight entirely (not entering another)
          if (!e.relatedTarget?.closest('.source-highlight')) {
            this.hideTooltip()
          }
        } else if (e.type === 'mousemove') {
          this.updateTooltipPosition(e)
        }
      }
      
      // Attach event listeners to container using event delegation
      container.addEventListener('mouseover', container._tooltipHandler, true)
      container.addEventListener('mouseout', container._tooltipHandler, true)
      container.addEventListener('mousemove', container._tooltipHandler)
    }
  },
  mounted() {
    this.$nextTick(() => {
      this.attachTooltipListeners()
    })
  },
  updated() {
    // Only re-attach if the container exists and doesn't have handlers yet
    // This prevents unnecessary re-attachment during tooltip position updates
    if (this.$refs.answerContent && !this.$refs.answerContent._tooltipHandler) {
      this.$nextTick(() => {
        this.attachTooltipListeners()
      })
    }
  },
  beforeDestroy() {
    // Clean up event listeners
    if (this.$refs.answerContent && this.$refs.answerContent._tooltipHandler) {
      const container = this.$refs.answerContent
      container.removeEventListener('mouseover', container._tooltipHandler)
      container.removeEventListener('mouseout', container._tooltipHandler)
      container.removeEventListener('mousemove', container._tooltipHandler)
      container._tooltipHandler = null
    }
  }
}
</script>

