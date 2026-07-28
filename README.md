# CORE 자동차 정비 예약 시스템

## 1. 프로젝트 개요
* 로컬 환경에서 독립적으로 실행되는 자동차 정비소 전용 예약 관리 시스템임.
* 작업 공간(베이) 단위의 스케줄링 및 중복 예약 방지 기능을 포함함.
* 외부 네트워크 연결 없이 사용 가능한 가벼운 SQLite 데이터베이스를 채택함.

## 2. 개발 스택
* **Backend**: Python 3.12, Flask, Flask-SQLAlchemy
* **Frontend**: HTML5, CSS3, JavaScript (ES6), Bootstrap 5, FullCalendar.js
* **Database**: SQLite
* **Document**: ReportLab (PDF 생성)

## 3. 기능 요약
1. **고객 예약 기능**: 달력 기반 10분 단위 타임슬롯, 점검 시간(03:00~05:00) 예약 차단, 다중 서비스 선택 및 소요 시간 자동 계산.
2. **관리자 기능**: 대시보드 통계, 예약 목록 확인 및 삭제, 서비스 항목(소요 시간, 식별 색상) 수정, 작업 공간 관리.
3. **출력물 지원**: 예약 리스트 CSV 다운로드 및 개별 작업지시서 PDF 자동 생성.

## 4. 설치 및 실행 방법
1. 필요 패키지 설치
   ```bash
   pip install -r requirements.txt