package openai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v7/sdk/config"
)

const (
	initialFailureChatModel   = "initial-failure-chat-model"
	initialHeartbeatChatModel = "initial-heartbeat-chat-model"
)

type delayedInitialStreamExecutor struct{}

func (*delayedInitialStreamExecutor) Identifier() string { return "delayed-initial-stream-executor" }

func (*delayedInitialStreamExecutor) Execute(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, errors.New("not implemented")
}

func (*delayedInitialStreamExecutor) ExecuteStream(ctx context.Context, _ *coreauth.Auth, _ coreexecutor.Request, _ coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	chunks := make(chan coreexecutor.StreamChunk)
	go func() {
		defer close(chunks)
		timer := time.NewTimer(1600 * time.Millisecond)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}
		select {
		case <-ctx.Done():
		case chunks <- coreexecutor.StreamChunk{Payload: []byte(`{"id":"delayed","choices":[{"delta":{"content":"hello"}}]}`)}:
		}
	}()
	return &coreexecutor.StreamResult{Chunks: chunks}, nil
}

func (*delayedInitialStreamExecutor) Refresh(_ context.Context, auth *coreauth.Auth) (*coreauth.Auth, error) {
	return auth, nil
}

func (*delayedInitialStreamExecutor) CountTokens(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, errors.New("not implemented")
}

func (*delayedInitialStreamExecutor) HttpRequest(context.Context, *coreauth.Auth, *http.Request) (*http.Response, error) {
	return nil, errors.New("not implemented")
}

type synchronizedResponseWriter struct {
	header http.Header
	mu     sync.Mutex
	status int
	body   bytes.Buffer
}

func newSynchronizedResponseWriter() *synchronizedResponseWriter {
	return &synchronizedResponseWriter{header: make(http.Header)}
}

func (w *synchronizedResponseWriter) Header() http.Header { return w.header }

func (w *synchronizedResponseWriter) WriteHeader(status int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.status == 0 {
		w.status = status
	}
}

func (w *synchronizedResponseWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.body.Write(p)
}

func (w *synchronizedResponseWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.status == 0 {
		w.status = http.StatusOK
	}
}

func (w *synchronizedResponseWriter) snapshot() (int, string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.status, w.body.String()
}

type initialFailureStreamExecutor struct{}

func (*initialFailureStreamExecutor) Identifier() string { return "initial-failure-stream-executor" }

func (*initialFailureStreamExecutor) Execute(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, errors.New("not implemented")
}

func (*initialFailureStreamExecutor) ExecuteStream(_ context.Context, _ *coreauth.Auth, _ coreexecutor.Request, _ coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	chunks := make(chan coreexecutor.StreamChunk, 1)
	chunks <- coreexecutor.StreamChunk{Err: errors.New("upstream failed before first payload")}
	close(chunks)
	return &coreexecutor.StreamResult{Chunks: chunks}, nil
}

func (*initialFailureStreamExecutor) Refresh(_ context.Context, auth *coreauth.Auth) (*coreauth.Auth, error) {
	return auth, nil
}

func (*initialFailureStreamExecutor) CountTokens(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, errors.New("not implemented")
}

func (*initialFailureStreamExecutor) HttpRequest(context.Context, *coreauth.Auth, *http.Request) (*http.Response, error) {
	return nil, errors.New("not implemented")
}

func runOpenAIStreamErrorTest(t *testing.T, endpoint string, body string) {
	gin.SetMode(gin.TestMode)

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			executor := &initialFailureStreamExecutor{}
			manager := coreauth.NewManager(nil, nil, nil)
			manager.RegisterExecutor(executor)
			authID := fmt.Sprintf("initial-failure-auth-%s-%d", strings.ReplaceAll(endpoint, "/", "-"), idx)
			auth := &coreauth.Auth{ID: authID, Provider: executor.Identifier(), Status: coreauth.StatusActive}
			if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
				t.Errorf("register auth %d: %v", idx, errRegister)
				return
			}
			registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: initialFailureChatModel}})
			defer registry.GetGlobalRegistry().UnregisterClient(auth.ID)

			base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager)
			h := NewOpenAIAPIHandler(base)
			router := gin.New()
			if endpoint == "/v1/chat/completions" {
				router.POST(endpoint, h.ChatCompletions)
			} else {
				router.POST(endpoint, h.Completions)
			}

			request := httptest.NewRequest(http.MethodPost, endpoint, strings.NewReader(body))
			request.Header.Set("Content-Type", "application/json")
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, request)

			if recorder.Code == http.StatusOK {
				t.Errorf("[%s] request %d lost the buffered initial error and returned HTTP 200: %q", endpoint, idx, recorder.Body.String())
			}
			if !strings.Contains(recorder.Body.String(), "upstream failed before first payload") {
				t.Errorf("[%s] request %d lost the initial upstream error: status=%d body=%q", endpoint, idx, recorder.Code, recorder.Body.String())
			}
		}(i)
	}
	wg.Wait()
}

func TestChatCompletionsHandlerDoesNotLoseErrorBeforeFirstPayload(t *testing.T) {
	runOpenAIStreamErrorTest(t, "/v1/chat/completions", `{"model":"initial-failure-chat-model","messages":[{"role":"user","content":"hi"}],"stream":true}`)
}

func TestCompletionsHandlerDoesNotLoseErrorBeforeFirstPayload(t *testing.T) {
	runOpenAIStreamErrorTest(t, "/v1/completions", `{"model":"initial-failure-chat-model","prompt":"hi","stream":true}`)
}

func TestWaitForInitialStreamEventReleasesHeartbeatBeforeFirstPayload(t *testing.T) {
	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage)
	heartbeat := make(chan time.Time, 1)
	heartbeat <- time.Unix(1, 0)

	event := waitForInitialStreamEvent(context.Background(), data, errs, heartbeat)
	if event.kind != initialStreamHeartbeat {
		t.Fatalf("waitForInitialStreamEvent() kind = %d, want heartbeat", event.kind)
	}
}

func TestWaitForInitialStreamEventKeepsImmediateErrorsAsHTTPFailures(t *testing.T) {
	data := make(chan []byte)
	errs := make(chan *interfaces.ErrorMessage, 1)
	want := &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: errors.New("initial failure")}
	errs <- want

	event := waitForInitialStreamEvent(context.Background(), data, errs, nil)
	if event.kind != initialStreamError || event.err != want {
		t.Fatalf("waitForInitialStreamEvent() = %#v, want initial error %#v", event, want)
	}
}

func TestChatCompletionsHandlerWritesHeartbeatBeforeDelayedFirstPayload(t *testing.T) {
	gin.SetMode(gin.TestMode)
	executor := &delayedInitialStreamExecutor{}
	manager := coreauth.NewManager(nil, nil, nil)
	manager.RegisterExecutor(executor)
	auth := &coreauth.Auth{ID: "initial-heartbeat-auth", Provider: executor.Identifier(), Status: coreauth.StatusActive}
	if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
		t.Fatalf("register auth: %v", errRegister)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: initialHeartbeatChatModel}})
	defer registry.GetGlobalRegistry().UnregisterClient(auth.ID)

	base := handlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{
		Streaming: sdkconfig.StreamingConfig{KeepAliveSeconds: 1},
	}, manager)
	h := NewOpenAIAPIHandler(base)
	router := gin.New()
	router.POST("/v1/chat/completions", h.ChatCompletions)
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"initial-heartbeat-chat-model","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	request.Header.Set("Content-Type", "application/json")
	writer := newSynchronizedResponseWriter()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(writer, request)
		close(done)
	}()

	heartbeatDeadline := time.NewTimer(1500 * time.Millisecond)
	defer heartbeatDeadline.Stop()
	heartbeatSeen := false
	for !heartbeatSeen {
		_, body := writer.snapshot()
		heartbeatSeen = strings.Contains(body, ": keep-alive\n\n")
		if heartbeatSeen {
			break
		}
		select {
		case <-heartbeatDeadline.C:
			t.Fatal("stream did not emit a heartbeat before the delayed first payload")
		case <-time.After(20 * time.Millisecond):
		}
	}

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not finish after delayed first payload")
	}
	status, body := writer.snapshot()
	if status != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%q", status, http.StatusOK, body)
	}
	if !strings.Contains(body, `"content":"hello"`) {
		t.Fatalf("stream body lost delayed payload: %q", body)
	}
}
