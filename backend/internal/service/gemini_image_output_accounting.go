package service

import (
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

// geminiImageOutputCounterKey 是请求级内联图片计数器挂在 gin.Context 上的键。
const geminiImageOutputCounterKey = "gemini_image_output_counter"

// geminiImageOutputCounter 记录一次转发里 Gemini 上游真正回吐的图片数量。
//
// 取「单个 payload 内的最大值」而不是累加：Gemini 兼容上游的 SSE 分片可能是
// 累积式的（同一份内容在后续 chunk 里重复整段回来——本文件同目录的
// computeGeminiTextDelta 就是为此存在的），逐 chunk 累加会把同一张图算很多次。
// 计费上宁可少算不可多算，所以用 max 兜底：
//   - 非流式：整份响应体只观测一次，max 即真实张数；
//   - 累积式流：最后一个 chunk 含全部图片，max 仍是真实张数；
//   - 增量式流且多图分散在不同 chunk：会低估到 1，与改动前的模型名启发式同值，
//     不构成回退。
type geminiImageOutputCounter struct {
	count int
}

// beginGeminiImageOutputObservation 在每次 Forward 开头重置计数器。
// failover 会拿同一个 gin.Context 重跑转发，不重置就会把上一个账号的图数带进来。
func beginGeminiImageOutputObservation(c *gin.Context) *geminiImageOutputCounter {
	if c == nil {
		return nil
	}
	counter := &geminiImageOutputCounter{}
	c.Set(geminiImageOutputCounterKey, counter)
	return counter
}

func geminiImageOutputCounterFromContext(c *gin.Context) *geminiImageOutputCounter {
	if c == nil {
		return nil
	}
	value, ok := c.Get(geminiImageOutputCounterKey)
	if !ok {
		return nil
	}
	counter, _ := value.(*geminiImageOutputCounter)
	return counter
}

// observeGeminiImageOutputs 观测一段上游响应（整份或单个 chunk）里的图片。
// 调用点与 upstreamResponseModelObserver.ObserveGemini 一一对应——那里拿得到
// 解包后的上游响应体，这里需要的是同一份字节。
func observeGeminiImageOutputs(c *gin.Context, payload []byte) {
	counter := geminiImageOutputCounterFromContext(c)
	if counter == nil {
		return
	}
	if count := countGeminiImageOutputs(payload); count > counter.count {
		counter.count = count
	}
}

func observedGeminiImageOutputs(c *gin.Context) int {
	counter := geminiImageOutputCounterFromContext(c)
	if counter == nil {
		return 0
	}
	return counter.count
}

// resolveGeminiImageCount 决定本次请求按几张图计费。
//
// 优先用上游真正返回的内联图片数：走 GeminiMessagesCompatService 的账号多是
// API Key + 自定义模型映射，客户端请求名和上游模型名都可能是站长自取的别名
// （issue #5358 里的 nana-banana-2），isImageGenerationModel 的白名单必然判不出，
// 于是 ImageCount=0，calculateRecordUsageCost 整条按次计费分支不触发，
// 生图请求全部记 $0。
//
// 只有响应里数不出图时（例如聚合函数丢掉了图片 part，或上游格式不受支持）
// 才退回既有的模型名启发式，保证老行为不回退；这里额外也认映射后的上游模型名，
// 与 shouldSkipCodexPlanGatedImageModelCooldown 对
// requestedModel / modelKey 双取的口径一致。
func resolveGeminiImageCount(c *gin.Context, originalModel, mappedModel string) int {
	if observed := observedGeminiImageOutputs(c); observed > 0 {
		return observed
	}
	if isImageGenerationModel(originalModel) || isImageGenerationModel(mappedModel) {
		return 1
	}
	return 0
}

// countGeminiImageOutputs 统计一段 Gemini 响应 JSON 里的图片 part。
// Gemini 可用 inlineData 提供图片内容，也可用 fileData 引用图片；两种形式
// 都可能以 camelCase 或 snake_case 返回。
func countGeminiImageOutputs(payload []byte) int {
	if len(payload) == 0 || !gjson.ValidBytes(payload) {
		return 0
	}
	count := 0
	gjson.GetBytes(payload, "candidates").ForEach(func(_, candidate gjson.Result) bool {
		candidate.Get("content.parts").ForEach(func(_, part gjson.Result) bool {
			if geminiPartIsImageOutput(part) {
				count++
			}
			return true
		})
		return true
	})
	return count
}

func geminiPartIsImageOutput(part gjson.Result) bool {
	return geminiImagePayloadIsPresent(part.Get("inlineData"), "data", "") ||
		geminiImagePayloadIsPresent(part.Get("inline_data"), "data", "") ||
		geminiImagePayloadIsPresent(part.Get("fileData"), "fileUri", "file_uri") ||
		geminiImagePayloadIsPresent(part.Get("file_data"), "fileUri", "file_uri")
}

func geminiImagePayloadIsPresent(payload gjson.Result, camelContentField, snakeContentField string) bool {
	if !payload.Exists() {
		return false
	}

	mimeType := payload.Get("mimeType")
	if !mimeType.Exists() {
		mimeType = payload.Get("mime_type")
	}
	if !isGeminiInlineImageMIMEType(strings.ToLower(strings.TrimSpace(mimeType.String()))) {
		return false
	}

	content := payload.Get(camelContentField)
	if !content.Exists() && snakeContentField != "" {
		content = payload.Get(snakeContentField)
	}
	// 空 Base64 数据或空文件 URI 不计费。
	return strings.TrimSpace(content.String()) != ""
}
