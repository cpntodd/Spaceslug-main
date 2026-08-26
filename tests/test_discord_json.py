import json, tempfile, unittest
from pathlib import Path
from spaceslug.workspace import WorkspaceService

class DiscordJsonTest(unittest.TestCase):
    def test_discordchat_export_is_structured(self):
        payload={"guild":{"name":"Direct Messages"},"channel":{"name":"Tester"},"messages":[{"id":"1","timestamp":"2024-01-01T00:00:00Z","content":"hello","author":{"name":"friend"}},{"id":"2","timestamp":"2024-01-01T00:00:01Z","content":"hi there","author":{"name":"cptn_oddsoul"}}]}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"chat.json"; p.write_text(json.dumps(payload),encoding="utf8")
            item=WorkspaceService(Path(d)/"ws").import_local(p)
            self.assertEqual(len(item.records),2)
            self.assertEqual(item.records[0]["author"],"friend")
            self.assertEqual(item.records[0]["conversation_role"],"user")
            self.assertEqual(item.records[1]["conversation_role"],"assistant")
            self.assertEqual(item.records[1]["content"],"hi there")
            self.assertIn("2024-01-01",item.records[1]["timestamp"])

if __name__ == '__main__': unittest.main()
