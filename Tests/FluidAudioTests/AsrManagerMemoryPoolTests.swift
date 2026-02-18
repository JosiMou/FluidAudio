import XCTest
@testable import FluidAudio

final class AsrManagerMemoryPoolTests: XCTestCase {
    override func tearDown() async throws {
        await sharedMLArrayCache.clear()
        try await super.tearDown()
    }

    func testExecuteInferenceReturnsPooledAudioArrayWhenInferenceThrows() async throws {
        await sharedMLArrayCache.clear()

        let manager = AsrManager()
        var decoderState = TdtDecoderState.make()
        let audio = Array(repeating: Float(0), count: 16_000)

        do {
            _ = try await manager.executeMLInferenceWithTimings(
                audio,
                originalLength: audio.count,
                decoderState: &decoderState
            )
            XCTFail("Expected notInitialized error.")
        } catch ASRError.notInitialized {
            // Expected.
        } catch {
            XCTFail("Expected ASRError.notInitialized, got \(error).")
        }

        let cachedCount = await sharedMLArrayCache.cachedArrayCount(
            shape: [NSNumber(value: 1), NSNumber(value: audio.count)],
            dataType: .float32
        )
        XCTAssertEqual(cachedCount, 1)
    }
}
