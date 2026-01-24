from datetime import datetime, timedelta


def convert_days_to_byday(days: str) -> str:
    map_ = {"M": "MO", "T": "TU", "W": "WE", "R": "TH", "F": "FR"}
    return ",".join(map_[d] for d in days.replace(" ", ""))


def convert_days_to_weekdays(days: str) -> list[int]:
    map_ = {"M": 1, "T": 2, "W": 3, "R": 4, "F": 5}
    return [map_[d] for d in days.replace(" ", "")]


def get_first_date_on_or_after(start_date: str, weekday: int) -> str:
    d = datetime.fromisoformat(start_date)
    diff = (weekday - d.isoweekday() + 7) % 7
    d = d + timedelta(days=diff)
    return d.date().isoformat()


def format_dt(date_str: str, time_str: str) -> str:
    time_24 = convert_to_24(time_str).replace(":", "")
    return (date_str + "T" + time_24 + "00").replace("-", "")


def format_until(end_date: str) -> str:
    return (end_date + "T235959").replace("-", "")


def convert_to_24(time12h: str) -> str:
    time, ampm = time12h.split(" ")
    h, m = map(int, time.split(":"))

    if ampm.lower() == "pm" and h != 12:
        h += 12
    if ampm.lower() == "am" and h == 12:
        h = 0

    return f"{h:02d}:{m:02d}"


def generate_ics(rows: list[dict], start_date: str, end_date: str, reminder: int) -> str:
    header = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//Class Schedule//EN\n"
        "CALSCALE:GREGORIAN\n"
        "X-WR-CALNAME:Classes\n"
    )
    footer = "END:VCALENDAR"

    events = []

    for idx, r in enumerate(rows):
        byday = convert_days_to_byday(r["days"])

        # DTSTART must be the FIRST occurrence on or after start_date
        first_weekday = convert_days_to_weekdays(r["days"])[0]
        first_date = get_first_date_on_or_after(start_date, first_weekday)

        start_dt = format_dt(first_date, r["start_time"])
        end_dt = format_dt(first_date, r["end_time"])

        rrule = f"FREQ=WEEKLY;BYDAY={byday};UNTIL={format_until(end_date)}"

        event = (
            "BEGIN:VEVENT\n"
            f"UID:{idx}-{int(datetime.now().timestamp())}@classschedule\n"
            f"SUMMARY:{r['course_code']}\n"
            f"LOCATION:{r['building']} {r['room']}\n"
            f"DTSTART:{start_dt}\n"
            f"DTEND:{end_dt}\n"
            f"RRULE:{rrule}\n"
            "BEGIN:VALARM\n"
            f"TRIGGER:-PT{reminder}M\n"
            "ACTION:DISPLAY\n"
            "DESCRIPTION:Reminder\n"
            "END:VALARM\n"
            "END:VEVENT\n"
        )

        events.append(event)

    return header + "".join(events) + footer
