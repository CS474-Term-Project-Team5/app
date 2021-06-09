class IssueHolder():
    def __init__(self, topic: list, num: int, person: list, org: list, place: list) -> None:
        self.topic = topic
        self.num = num
        self.person = person
        self.org = org
        self.place = place
        self.event = None

    def __init__(self, topic: list, num: int, vec) -> None:
        self.topic = topic
        self.num = num
        self.event = None
        self.vec = vec

    def add_event(self, event: list):
        self.event = event
